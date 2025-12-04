import argparse
import os
import sys
import cv2
import numpy as np
import yaml
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

def image2np(msg):
    """Converts ROS image message to numpy array."""
    h, w = msg.height, msg.width
    enc = (msg.encoding or "").lower()
    buf = memoryview(msg.data)
    
    if enc in ("mono8", "8uc1"):
        return np.frombuffer(buf, dtype=np.uint8).reshape(h, w)
    elif enc in ("mono16", "16uc1", "16sc1"):
        return np.frombuffer(buf, dtype=np.uint16).reshape(h, w)
    elif enc == "bgr8":
        return np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
    elif enc == "rgb8":
        # Convert RGB to BGR for OpenCV
        img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif enc == "bgra8":
        return np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
    elif enc == "rgba8":
        img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    else:
        # Try to handle as mono8 if unknown, or raise error
        print(f"Warning: Unsupported encoding {enc}, trying as mono8")
        return np.frombuffer(buf, dtype=np.uint8).reshape(h, w)

def process_image(img, options):
    """Applies requested processing to the image based on dictionary options."""
    
    # 1. Convert to Grayscale if requested
    if options.get('grayscale', False):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
    # 2. Bit depth conversion (16-bit to 8-bit)
    if options.get('to_8bit', False):
        if img.dtype == np.uint16:
            # Normalize to 0-255 range based on min/max of the image
            min_val = np.min(img)
            max_val = np.max(img)
            if max_val > min_val:
                img = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                img = np.zeros_like(img, dtype=np.uint8)
        elif img.dtype != np.uint8:
             # Fallback for floats etc
             img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 3. Normalization (Min-Max) - if not already done by to_8bit or if requested explicitly for 8-bit
    if options.get('norm', False) and not options.get('to_8bit', False): 
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        if img.dtype != np.uint8:
             img = img.astype(np.uint8)

    # 4. CLAHE
    if options.get('clahe', False):
        # CLAHE only works on single channel (grayscale) or L channel of LAB
        if len(img.shape) == 2: # Grayscale
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img = clahe.apply(img)
        elif len(img.shape) == 3:
            # Convert to LAB, apply to L, convert back
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            lab = cv2.merge((l,a,b))
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
    return img

def main():
    parser = argparse.ArgumentParser(description="Extract and process images from ROS bag using YAML config.")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file")
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    bag_file = config.get('bag_file')
    if not bag_file:
        print("Error: 'bag_file' must be specified in config.")
        sys.exit(1)
        
    output_dir = config.get('output_dir', 'output_images')
    # Use bag filename (without extension) as subfolder
    output_dir = os.path.join(output_dir, Path(bag_file).stem)
    sync = config.get('sync', False)
    time_tol = config.get('time_tol', 0.05)
    start_index = config.get('start_index', 0)
    end_index = config.get('end_index', float('inf'))
    topics_config = config.get('topics', {})
    
    if not topics_config:
        print("Error: No 'topics' defined in config.")
        sys.exit(1)
        
    target_topics = list(topics_config.keys())
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirs for topics
    topic_dirs = {}
    safe_topic_names = [t.replace('/', '_').strip('_') for t in target_topics]
    for t, safe_name in zip(target_topics, safe_topic_names):
        path = os.path.join(output_dir, safe_name)
        os.makedirs(path, exist_ok=True)
        topic_dirs[t] = path
        
    print(f"Processing bag: {bag_file}")
    print(f"Topics: {target_topics}")
    print(f"Filtering index: [{start_index}, {end_index}]")
    
    with AnyReader([Path(bag_file)]) as reader:
        if sync:
            print("Extraction with sync enabled...")
            
            # Buffer all messages
            msgs_by_topic = {t: [] for t in target_topics}
            
            print("Reading messages...")
            connections = [x for x in reader.connections if x.topic in target_topics]
            for connection, timestamp, rawdata in reader.messages(connections=connections):
                msg = reader.deserialize(rawdata, connection.msgtype)
                t_sec = timestamp / 1e9
                msgs_by_topic[connection.topic].append((t_sec, msg))
                
            # Sort
            for t in target_topics:
                msgs_by_topic[t].sort(key=lambda x: x[0])
                
            ref_topic = target_topics[0]
            ref_msgs = msgs_by_topic[ref_topic]
            
            print(f"Syncing based on {ref_topic} ({len(ref_msgs)} msgs)...")
            
            count = 0
            saved_count = 0
            for ref_time, ref_msg in ref_msgs:
                current_set = {ref_topic: ref_msg}
                match_found = True
                
                for other_topic in target_topics[1:]:
                    candidates = msgs_by_topic[other_topic]
                    if not candidates:
                        match_found = False
                        break
                        
                    nearby = [m for m in candidates if abs(m[0] - ref_time) <= time_tol]
                    
                    if not nearby:
                        match_found = False
                        break
                        
                    best_match = min(nearby, key=lambda x: abs(x[0] - ref_time))
                    current_set[other_topic] = best_match[1]
                    
                if match_found:
                    # Use sequential numbering for synced images as requested
                    # count starts at 0, so we use count + 1 for 1-based indexing
                    seq_id = count + 1
                    
                    # Check index range
                    if start_index <= seq_id <= end_index:
                        for topic, msg in current_set.items():
                            try:
                                cv_img = image2np(msg)
                                # Get topic specific options
                                opts = topics_config.get(topic, {})
                                cv_img = process_image(cv_img, opts)
                                
                                fname = f"{seq_id}.png"
                                save_path = os.path.join(topic_dirs[topic], fname)
                                cv2.imwrite(save_path, cv_img)
                            except Exception as e:
                                print(f"Error processing {topic}: {e}")
                        saved_count += 1
                        
                    count += 1
                    if count % 10 == 0:
                        print(f"Processed {count} synced sets (Saved {saved_count})...", end='\r')
                        
            print(f"\nTotal synced sets processed: {count}")
            print(f"Total synced sets saved: {saved_count}")
            
        else:
            print("Extraction without sync...")
            count = 0
            saved_count = 0
            connections = [x for x in reader.connections if x.topic in target_topics]
            for connection, timestamp, rawdata in reader.messages(connections=connections):
                try:
                    # Use sequential indexing for unsynced mode too
                    count += 1
                    
                    # Check index range
                    if start_index <= count <= end_index:
                        msg = reader.deserialize(rawdata, connection.msgtype)
                        cv_img = image2np(msg)
                        
                        # Get topic specific options
                        opts = topics_config.get(connection.topic, {})
                        cv_img = process_image(cv_img, opts)
                        
                        fname = f"{count}.png"
                        save_path = os.path.join(topic_dirs[connection.topic], fname)
                        cv2.imwrite(save_path, cv_img)
                        saved_count += 1
                    
                    if count % 50 == 0:
                        print(f"Processed {count} images (Saved {saved_count})...", end='\r')
                except Exception as e:
                    print(f"Error processing {connection.topic}: {e}")
                    
            print(f"\nTotal images processed: {count}")
            print(f"Total images saved: {saved_count}")

if __name__ == "__main__":
    main()
