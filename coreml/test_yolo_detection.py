#!/usr/bin/env python3
import cv2
import coremltools as ct
import numpy as np
import os
from PIL import Image

def preprocess_frame_for_yolo(frame, target_size=640):
    """Preprocess frame for YOLO model input."""
    h, w = frame.shape[:2]
    
    # Calculate scaling and padding to maintain aspect ratio
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Resize frame
    resized = cv2.resize(frame, (new_w, new_h))
    
    # Create padded frame
    padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    # Calculate padding offsets to center the image
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    
    padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    # Convert BGR to RGB for PIL
    padded_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(padded_rgb)
    
    return pil_image, scale, x_offset, y_offset

def detect_persons_simple(frame, yolo_model, confidence_threshold=0.5):
    """Detect persons in frame using YOLO model."""
    # Preprocess frame
    input_data, scale, x_offset, y_offset = preprocess_frame_for_yolo(frame)
    
    print(f"YOLO input size: {input_data.size}")
    print(f"Scale: {scale}, Offsets: ({x_offset}, {y_offset})")
    
    # Run YOLO inference
    try:
        prediction = yolo_model.predict({"image": input_data})
        print(f"YOLO prediction keys: {list(prediction.keys())}")
        
        # Get the output (typically named 'output' or similar)
        output_name = list(prediction.keys())[0]
        detections = prediction[output_name]
        print(f"YOLO output shape: {detections.shape}")
        print(f"YOLO output range: [{detections.min():.3f}, {detections.max():.3f}]")
        
        # Parse detections - YOLO11 format appears to be [batch, 84, 8400]
        # where 84 = 4 (bbox) + 80 (classes), and 8400 is the number of detection candidates
        if len(detections.shape) == 3:
            detections = detections[0]  # Remove batch dimension, now (84, 8400)
        
        print(f"Detections shape after batch removal: {detections.shape}")
        
        # Transpose to get (8400, 84) format for easier processing
        detections = detections.T  # Now (8400, 84)
        print(f"Detections shape after transpose: {detections.shape}")
        
        person_boxes = []
        h, w = frame.shape[:2]
        
        # Process each detection candidate
        for i, detection in enumerate(detections):
            if len(detection) >= 84:  # YOLO11 format: 4 bbox + 80 classes
                # Extract bbox coordinates (first 4 values)
                x_center, y_center, width, height = detection[:4]
                
                # Extract class scores (remaining 80 values)
                class_scores = detection[4:84]
                
                # Person class is typically index 0 in COCO dataset
                person_score = class_scores[0]
                
                if person_score > confidence_threshold:
                    # Convert from normalized coordinates (0-640) to pixel coordinates
                    # Account for preprocessing scaling and padding
                    x_center_orig = (x_center - x_offset) / scale
                    y_center_orig = (y_center - y_offset) / scale
                    width_orig = width / scale
                    height_orig = height / scale
                    
                    # Convert to top-left corner format
                    x1 = int(x_center_orig - width_orig / 2)
                    y1 = int(y_center_orig - height_orig / 2)
                    x2 = int(x_center_orig + width_orig / 2)
                    y2 = int(y_center_orig + height_orig / 2)
                    
                    # Clamp to image bounds
                    x1 = max(0, min(x1, w))
                    y1 = max(0, min(y1, h))
                    x2 = max(0, min(x2, w))
                    y2 = max(0, min(y2, h))
                    
                    # Only keep valid boxes
                    if x2 > x1 and y2 > y1:
                        person_boxes.append((x1, y1, x2, y2, person_score))
                        print(f"Found person: bbox=({x1}, {y1}, {x2}, {y2}), conf={person_score:.3f}")
        
        print(f"Total person detections before NMS: {len(person_boxes)}")
        
        # Apply Non-Maximum Suppression to remove duplicate detections
        if len(person_boxes) > 0:
            # Convert to format needed for cv2.dnn.NMSBoxes
            boxes = [(x1, y1, x2-x1, y2-y1) for x1, y1, x2, y2, conf in person_boxes]  # (x, y, w, h)
            confidences = [conf for x1, y1, x2, y2, conf in person_boxes]
            
            # Apply NMS
            indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.5)  # IoU threshold = 0.5
            
            # Filter detections based on NMS results
            if len(indices) > 0:
                person_boxes = [person_boxes[i] for i in indices.flatten()]
        
        print(f"Total person detections after NMS: {len(person_boxes)}")
        return person_boxes
        
    except Exception as e:
        print(f"YOLO inference error: {e}")
        import traceback
        traceback.print_exc()
        return []

def test_yolo_on_video(video_path, yolo_model_path, max_frames=10, confidence_threshold=0.5):
    """Test YOLO detection on video frames."""
    print(f"Testing YOLO detection on: {video_path}")
    print(f"Using model: {yolo_model_path}")
    print(f"Confidence threshold: {confidence_threshold}")
    print("=" * 50)
    
    # Load YOLO model
    if not os.path.exists(yolo_model_path):
        print(f"Error: YOLO model not found: {yolo_model_path}")
        return
    
    try:
        print("Loading YOLO model...")
        yolo_model = ct.models.MLModel(yolo_model_path)
        print("YOLO model loaded successfully!")
        print(f"Input description: {yolo_model.input_description}")
        print(f"Output description: {yolo_model.output_description}")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video: {video_path}")
        return
    
    # Get video info
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo info: {width}x{height} @ {fps}fps, {total_frames} total frames")
    print("-" * 50)
    
    frame_count = 0
    total_detections = 0
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {frame_count + 1}")
            break
        
        frame_count += 1
        print(f"\nProcessing frame {frame_count}...")
        
        # Detect persons
        person_boxes = detect_persons_simple(frame, yolo_model, confidence_threshold)
        total_detections += len(person_boxes)
        
        # Draw bounding boxes
        frame_with_boxes = frame.copy()
        for x1, y1, x2, y2, conf in person_boxes:
            cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_with_boxes, f"Person {conf:.2f}", 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Save frame with detections
        output_filename = f"yolo_detection_frame_{frame_count:03d}.jpg"
        cv2.imwrite(output_filename, frame_with_boxes)
        print(f"Saved: {output_filename}")
    
    cap.release()
    
    print("\n" + "=" * 50)
    print(f"YOLO Detection Test Complete!")
    print(f"Processed {frame_count} frames")
    print(f"Total person detections: {total_detections}")
    print(f"Average detections per frame: {total_detections / frame_count:.1f}")

if __name__ == "__main__":
    # Test parameters
    video_path = "../data/cam01.mp4"
    yolo_model_path = "models/yolo11n.mlpackage"
    
    # Run test
    test_yolo_on_video(video_path, yolo_model_path, max_frames=5, confidence_threshold=0.3)