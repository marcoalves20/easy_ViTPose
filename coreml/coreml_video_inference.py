#!/usr/bin/env python3
import coremltools as ct
import numpy as np
from PIL import Image
import cv2
import os
import argparse

def preprocess_frame_for_yolo(frame, target_size=640):
    """
    Preprocess frame for YOLO detection model.
    Args:
        frame: OpenCV frame (BGR format)
        target_size: Target size for YOLO (square input)
    Returns:
        PIL Image and scale info for YOLO inference
    """
    # Get original dimensions
    h, w = frame.shape[:2]
    
    # Calculate scale to fit into square
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize frame
    resized = cv2.resize(frame, (new_w, new_h))
    
    # Create square canvas and center the image
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    # Convert BGR to RGB for PIL
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image (required for CoreML YOLO model)
    pil_image = Image.fromarray(canvas_rgb)
    
    return pil_image, scale, x_offset, y_offset

def detect_persons(frame, yolo_model, confidence_threshold=0.5):
    """
    Detect persons in frame using YOLO model.
    Args:
        frame: OpenCV frame (BGR format)
        yolo_model: CoreML YOLO model
        confidence_threshold: Minimum confidence for detection
    Returns:
        List of person bounding boxes [(x1, y1, x2, y2, confidence), ...]
    """
    h, w = frame.shape[:2]
    
    # Preprocess frame for YOLO
    input_data, scale, x_offset, y_offset = preprocess_frame_for_yolo(frame, 640)
    
    # Run YOLO inference
    try:
        prediction = yolo_model.predict({"image": input_data})
        
        # Get output - YOLO11 format is [batch, 84, 8400] 
        # where 84 = 4 (bbox) + 80 (classes), and 8400 is the number of detection candidates
        output_key = list(prediction.keys())[0]
        detections = prediction[output_key]  # Shape: (1, 84, 8400)
        
        if len(detections.shape) == 3:
            detections = detections[0]  # Remove batch dimension: (84, 8400)
        
        # Transpose to get (8400, 84) format for easier processing
        detections = detections.T  # Now (8400, 84)
        
        person_boxes = []
        
        # Process each detection candidate
        for detection in detections:
            if len(detection) >= 84:  # YOLO11 format: 4 bbox + 80 classes
                # Extract bbox coordinates (first 4 values)
                center_x, center_y, bbox_w, bbox_h = detection[:4]
                
                # Extract class scores (remaining 80 values)
                class_scores = detection[4:84]
                
                # Person class is typically index 0 in COCO dataset
                person_conf = class_scores[0]
                
                if person_conf > confidence_threshold:
                    # Convert from normalized coordinates (0-640) to pixel coordinates
                    # Account for preprocessing scaling and padding
                    center_x_orig = (center_x - x_offset) / scale
                    center_y_orig = (center_y - y_offset) / scale
                    bbox_w_orig = bbox_w / scale
                    bbox_h_orig = bbox_h / scale
                    
                    # Convert to corner coordinates
                    x1 = max(0, int(center_x_orig - bbox_w_orig/2))
                    y1 = max(0, int(center_y_orig - bbox_h_orig/2))
                    x2 = min(w, int(center_x_orig + bbox_w_orig/2))
                    y2 = min(h, int(center_y_orig + bbox_h_orig/2))
                    
                    # Only add valid boxes
                    if x2 > x1 and y2 > y1:
                        person_boxes.append((x1, y1, x2, y2, person_conf))
        
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
        
        return person_boxes
        
    except Exception as e:
        print(f"YOLO detection failed: {e}")
        return []

def preprocess_frame(frame, target_size=(192, 256)):
    """
    Preprocess video frame for ViTPose model.
    Args:
        frame: OpenCV frame (BGR format)
        target_size: Target size (width, height)
    Returns:
        Preprocessed frame array
    """
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image for resizing
    image = Image.fromarray(frame_rgb)
    
    # Resize to target size
    image = image.resize(target_size, Image.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(image, dtype=np.float32)
    
    # Normalize to [0, 1]
    img_array = img_array / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    
    # Convert from HWC to CHW format
    img_array = np.transpose(img_array, (2, 0, 1))
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def postprocess_heatmaps(heatmaps, original_size, input_size=(192, 256)):
    """
    Convert heatmaps to 2D keypoint coordinates.
    Args:
        heatmaps: Model output heatmaps (1, 25, 64, 48)
        original_size: Original frame size (width, height)
        input_size: Model input size (width, height)
    Returns:
        keypoints: Array of (x, y, confidence) for each keypoint
    """
    # Remove batch dimension
    heatmaps = heatmaps[0]  # (25, 64, 48)
    
    keypoints = []
    
    for i in range(heatmaps.shape[0]):  # For each of the 25 keypoints
        heatmap = heatmaps[i]  # (64, 48)
        
        # Find the maximum value and its location
        max_val = np.max(heatmap)
        max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        
        # Convert heatmap coordinates to input image coordinates
        y_heatmap, x_heatmap = max_idx
        
        # Scale from heatmap resolution (64, 48) to input resolution (256, 192)
        x_input = x_heatmap * (input_size[1] / heatmap.shape[1])  # 192 / 48
        y_input = y_heatmap * (input_size[0] / heatmap.shape[0])  # 256 / 64
        
        # Scale from input resolution to original frame resolution
        x_original = x_input * (original_size[0] / input_size[1])
        y_original = y_input * (original_size[1] / input_size[0])
        
        keypoints.append([x_original, y_original, max_val])
    
    return np.array(keypoints)

def draw_keypoints_on_frame(frame, keypoints, confidence_threshold=0.5):
    """
    Draw keypoints and skeleton on a video frame.
    Args:
        frame: OpenCV frame (BGR format)
        keypoints: Array of (x, y, confidence) for each keypoint
        confidence_threshold: Minimum confidence to draw keypoints
    Returns:
        frame: Frame with keypoints and skeleton drawn
    """
    # Keypoint names for reference
    keypoint_names = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'neck', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
        'left_big_toe', 'left_small_toe', 'left_heel',
        'right_big_toe', 'right_small_toe', 'right_heel'
    ]
    
    # Define skeleton connections
    skeleton = [
        # Head connections
        (0, 1), (0, 2),           # nose to eyes
        (1, 3), (2, 4),           # eyes to ears
        (0, 5),                   # nose to neck
        
        # Upper body connections
        (5, 6), (5, 7),           # neck to shoulders
        (6, 8), (7, 9),           # shoulders to elbows
        (8, 10), (9, 11),         # elbows to wrists
        
        # Torso connections
        (5, 14),                  # neck to hip center
        (6, 12), (7, 13),         # shoulders to hips
        (12, 14), (13, 14),       # hips to hip center
        
        # Lower body connections
        (12, 15), (13, 16),       # hips to knees
        (15, 17), (16, 18),       # knees to ankles
        
        # Foot connections
        (17, 19), (17, 20), (17, 21),  # left ankle to toes/heel
        (18, 22), (18, 23), (18, 24),  # right ankle to toes/heel
        (19, 20), (22, 23),            # big toe to small toe
    ]
    
    # Draw skeleton connections
    for connection in skeleton:
        pt1_idx, pt2_idx = connection
        if (pt1_idx < len(keypoints) and pt2_idx < len(keypoints) and
            keypoints[pt1_idx][2] > confidence_threshold and 
            keypoints[pt2_idx][2] > confidence_threshold):
            
            pt1 = (int(keypoints[pt1_idx][0]), int(keypoints[pt1_idx][1]))
            pt2 = (int(keypoints[pt2_idx][0]), int(keypoints[pt2_idx][1]))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
    
    # Draw keypoints
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > confidence_threshold:
            # Use different colors for different keypoint types
            if i < 5:  # Head keypoints (0-4: nose, eyes, ears)
                color = (0, 0, 255)  # Red
            elif i == 5:  # Neck
                color = (255, 255, 0)  # Cyan
            elif i < 12:  # Arm keypoints (6-11: shoulders, elbows, wrists)
                color = (255, 0, 0)  # Blue
            elif i < 15:  # Hip keypoints (12-14: hips, pelvis)
                color = (128, 0, 128)  # Purple
            elif i < 19:  # Leg keypoints (15-18: knees, ankles)
                color = (0, 255, 255)  # Yellow
            else:  # Foot keypoints (19-24: toes, heels)
                color = (0, 128, 255)  # Orange
            
            # Draw keypoint with outline for visibility
            cv2.circle(frame, (int(x), int(y)), 5, color, -1)
            cv2.circle(frame, (int(x), int(y)), 7, (255, 255, 255), 2)
    
    return frame

def process_video_with_detection(input_path, output_path, pose_model_path="ViTPose_PyTorch.mlpackage", 
                               yolo_model_path="yolo11n.mlpackage", confidence_threshold=0.5, 
                               detection_threshold=0.6):
    """
    Process video file with person detection and pose estimation.
    Args:
        input_path: Path to input video
        output_path: Path to output video  
        pose_model_path: Path to ViTPose CoreML model
        yolo_model_path: Path to YOLO detection model
        confidence_threshold: Minimum confidence for keypoint visualization
        detection_threshold: Minimum confidence for person detection
    """
    print(f"Processing video: {input_path}")
    
    # Load CoreML models
    print("Loading ViTPose model...")
    pose_model = ct.models.MLModel(pose_model_path)
    
    print("Loading YOLO detection model...")
    yolo_model = ct.models.MLModel(yolo_model_path)
    
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Setup output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_persons_detected = 0
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            print(f"Processing frame {frame_count}/{total_frames}", end='\r')
            
            # Create a copy for drawing
            frame_with_poses = frame.copy()
            
            try:
                # Step 1: Detect persons using YOLO
                person_boxes = detect_persons(frame, yolo_model, detection_threshold)
                
                frame_person_count = len(person_boxes)
                total_persons_detected += frame_person_count
                
                # Step 2: Run pose estimation on each detected person
                for i, (x1, y1, x2, y2, det_conf) in enumerate(person_boxes):
                    # Expand bounding box slightly for better pose estimation
                    margin = 20
                    x1_exp = max(0, x1 - margin)
                    y1_exp = max(0, y1 - margin) 
                    x2_exp = min(width, x2 + margin)
                    y2_exp = min(height, y2 + margin)
                    
                    # Draw detection bounding box
                    cv2.rectangle(frame_with_poses, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame_with_poses, f"Person {i+1}: {det_conf:.2f}", 
                               (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    
                    # Crop person region
                    person_crop = frame[y1_exp:y2_exp, x1_exp:x2_exp]
                    
                    if person_crop.size > 0:
                        # Preprocess cropped region for pose estimation
                        crop_input = preprocess_frame(person_crop)
                        
                        # Run pose estimation on crop
                        # Get input/output names from model spec
                        pose_spec = pose_model._spec
                        pose_input_name = pose_spec.description.input[0].name
                        pose_output_name = pose_spec.description.output[0].name
                        
                        pose_prediction = pose_model.predict({pose_input_name: crop_input})
                        heatmaps = pose_prediction[pose_output_name]
                        
                        # Get keypoints in crop coordinates
                        crop_height, crop_width = person_crop.shape[:2]
                        crop_keypoints = postprocess_heatmaps(heatmaps, 
                                                            original_size=(crop_width, crop_height))
                        
                        # Transform keypoints back to full frame coordinates
                        frame_keypoints = []
                        for kp in crop_keypoints:
                            x_frame = kp[0] + x1_exp
                            y_frame = kp[1] + y1_exp
                            confidence = kp[2]
                            frame_keypoints.append([x_frame, y_frame, confidence])
                        
                        frame_keypoints = np.array(frame_keypoints)
                        
                        # Draw keypoints and skeleton on full frame
                        frame_with_poses = draw_keypoints_on_frame(frame_with_poses, 
                                                                 frame_keypoints, 
                                                                 confidence_threshold)
                
                # Add frame statistics
                cv2.putText(frame_with_poses, f"Frame: {frame_count}/{total_frames}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame_with_poses, f"Persons: {frame_person_count}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Write frame to output video
                out.write(frame_with_poses)
                
                # Debug info every 20 frames
                if frame_count % 20 == 0:
                    avg_persons = total_persons_detected / frame_count
                    print(f"\nFrame {frame_count}: {frame_person_count} persons, avg: {avg_persons:.1f}")
                
            except Exception as e:
                print(f"\nError processing frame {frame_count}: {e}")
                import traceback
                traceback.print_exc()
                # Write original frame if inference fails
                out.write(frame)
    
    except KeyboardInterrupt:
        print(f"\nInterrupted at frame {frame_count}")
    
    finally:
        # Clean up
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        avg_persons = total_persons_detected / frame_count if frame_count > 0 else 0
        print(f"\nVideo processing complete!")
        print(f"Output saved to: {output_path}")
        print(f"Processed {frame_count} frames")
        print(f"Total persons detected: {total_persons_detected}")
        print(f"Average persons per frame: {avg_persons:.1f}")

def process_video(input_path, output_path, model_path="ViTPose_PyTorch.mlpackage", confidence_threshold=0.5):
    """Legacy function for backward compatibility - runs pose estimation on full frame."""
    return process_video_with_detection(input_path, output_path, model_path,
                                        "models/yolo11n.mlpackage", confidence_threshold)

def main():
    parser = argparse.ArgumentParser(description="ViTPose video inference with YOLO detection and CoreML")
    parser.add_argument("input_video", help="Path to input video file")
    parser.add_argument("-o", "--output", default="output_video_with_pose.mp4", 
                       help="Output video path (default: output_video_with_pose.mp4)")
    parser.add_argument("-m", "--model", default="ViTPose_PyTorch.mlpackage",
                       help="Path to ViTPose CoreML model (default: ViTPose_PyTorch.mlpackage)")
    parser.add_argument("-y", "--yolo", default="yolo11n.mlpackage",
                       help="Path to YOLO detection model (default: yolo11n.mlpackage)")
    parser.add_argument("-c", "--confidence", type=float, default=0.5,
                       help="Confidence threshold for keypoints (default: 0.5)")
    parser.add_argument("-d", "--detection", type=float, default=0.6,
                       help="Detection confidence threshold for YOLO (default: 0.6)")
    parser.add_argument("--no-detection", action="store_true",
                       help="Skip person detection, run pose estimation on full frame")
    
    args = parser.parse_args()
    
    # Check if input video exists
    if not os.path.exists(args.input_video):
        print(f"Error: Input video '{args.input_video}' not found")
        return
    
    # Check if models exist
    if not os.path.exists(args.model):
        print(f"Error: ViTPose model '{args.model}' not found")
        return
    
    if not args.no_detection and not os.path.exists(args.yolo):
        print(f"Error: YOLO model '{args.yolo}' not found")
        print("Use --no-detection to skip person detection")
        return
    
    try:
        if args.no_detection:
            # Use legacy full-frame pose estimation
            print("Running pose estimation on full frames (no person detection)")
            from coreml_inference import run_inference, visualize_keypoints
            # For now, fall back to old method
            print("Full-frame mode not yet implemented in video processor")
            return
        else:
            # Use new detection + pose estimation pipeline
            print(f"Using YOLO detection + ViTPose estimation pipeline")
            print(f"Detection threshold: {args.detection}, Pose threshold: {args.confidence}")
            
            process_video_with_detection(
                args.input_video, 
                args.output, 
                args.model, 
                args.yolo,
                args.confidence,
                args.detection
            )
            
        print(f"\n✅ Success! Output video saved as '{args.output}'")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # If no command line arguments, use default test setup
    import sys
    if len(sys.argv) == 1:
        print("ViTPose Video Inference")
        print("=" * 30)
        print("Usage: python coreml_video_inference.py <input_video> [options]")
        print("\nExample:")
        print("  python coreml_video_inference.py test_video.mp4 -o result.mp4")
        print("  python coreml_video_inference.py data/cam01.mp4 -c 0.3 -d 0.5")
        print("\nOptions:")
        print("  -o, --output      Output video path")
        print("  -m, --model       ViTPose CoreML model path (default: ViTPose_PyTorch.mlpackage)")
        print("  -y, --yolo        YOLO detection model path (default: yolo11n.mlpackage)")
        print("  -c, --confidence  Keypoint confidence threshold (0.0-1.0)")
        print("  -d, --detection   Person detection threshold (0.0-1.0)")
        print("  --no-detection    Skip person detection, run pose on full frame")
        print("\nFor help: python coreml_video_inference.py --help")
    else:
        main()