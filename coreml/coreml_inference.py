#!/usr/bin/env python3
import coremltools as ct
import numpy as np
from PIL import Image
import cv2

def preprocess_image(image_path, target_size=(192, 256)):
    """
    Preprocess image for ViTPose model.
    Args:
        image_path: Path to input image
        target_size: Target size (width, height)
    Returns:
        Preprocessed image array
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
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

def run_inference():
    """Run inference on test image using CoreML model."""
    
    # Load the CoreML model
    print("Loading CoreML model...")
    model = ct.models.MLModel("ViTPose_PyTorch.mlpackage")
    
    # Print model info
    print("\nModel Information:")
    try:
        print(f"Input description: {model.input_description}")
        print(f"Output description: {model.output_description}")
    except:
        # For our converted model, get the info from spec
        spec = model._spec
        input_info = spec.description.input[0]
        output_info = spec.description.output[0]
        print(f"Input: {input_info.name} {[d.dim_value if hasattr(d, 'dim_value') else d for d in input_info.type.multiArrayType.shape]}")
        print(f"Output: {output_info.name} {[d.dim_value if hasattr(d, 'dim_value') else d for d in output_info.type.multiArrayType.shape]}")
    
    # Load and preprocess test image
    image_path = "../data/test_frame_1.jpg"
    print(f"\nPreprocessing image: {image_path}")
    
    # Preprocess image
    input_data = preprocess_image(image_path)
    print(f"Input shape: {input_data.shape}")
    print(f"Input range: [{input_data.min():.3f}, {input_data.max():.3f}]")
    
    # Run inference
    print("\nRunning inference...")
    
    # Get the input name from the model spec
    spec = model._spec
    input_name = spec.description.input[0].name
    print(f"Using input name: {input_name}")
    
    # Create input dictionary
    input_dict = {input_name: input_data}
    
    # Run prediction
    try:
        prediction = model.predict(input_dict)
        print("Inference successful!")
        
        # Print output information
        for output_name, output_value in prediction.items():
            print(f"Output '{output_name}' shape: {output_value.shape}")
            print(f"Output '{output_name}' range: [{output_value.min():.3f}, {output_value.max():.3f}]")
        
        return prediction
        
    except Exception as e:
        print(f"Inference failed: {e}")
        return None

def postprocess_heatmaps(heatmaps, original_size=(408, 612), input_size=(192, 256)):
    """
    Convert heatmaps to 2D keypoint coordinates.
    Args:
        heatmaps: Model output heatmaps (1, 25, 64, 48)
        original_size: Original image size (width, height)
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
        
        # Scale from input resolution to original image resolution
        x_original = x_input * (original_size[0] / input_size[1])  # 408 / 192
        y_original = y_input * (original_size[1] / input_size[0])  # 612 / 256
        
        keypoints.append([x_original, y_original, max_val])
    
    return np.array(keypoints)

def visualize_keypoints(image_path, predictions, output_path="output_with_keypoints.jpg"):
    """
    Visualize keypoints on the original image.
    Args:
        image_path: Path to original image
        predictions: Model predictions dictionary
        output_path: Path to save visualization
    """
    # Load original image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    # Get image dimensions
    h, w = image.shape[:2]
    print(f"Original image size: {w}x{h}")
    
    # Get heatmaps from predictions
    output_name = list(predictions.keys())[0]
    heatmaps = predictions[output_name]
    
    # Post-process heatmaps to get keypoint coordinates
    keypoints = postprocess_heatmaps(heatmaps, original_size=(w, h))
    
    # Correct keypoint names for your model
    keypoint_names = [
        'nose',           # 0
        'left_eye',       # 1
        'right_eye',      # 2
        'left_ear',       # 3
        'right_ear',      # 4
        'neck',           # 5
        'left_shoulder',  # 6
        'right_shoulder', # 7
        'left_elbow',     # 8
        'right_elbow',    # 9
        'left_wrist',     # 10
        'right_wrist',    # 11
        'left_hip',       # 12
        'right_hip',      # 13
        'hip',            # 14 (pelvis/root)
        'left_knee',      # 15
        'right_knee',     # 16
        'left_ankle',     # 17
        'right_ankle',    # 18
        'left_big_toe',   # 19
        'left_small_toe', # 20
        'left_heel',      # 21
        'right_big_toe',  # 22
        'right_small_toe',# 23
        'right_heel'      # 24
    ]
    
    # Define skeleton connections based on human anatomy
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
    
    # Confidence threshold - lowered to see more keypoints
    confidence_threshold = 0.05
    
    # Draw skeleton connections
    for connection in skeleton:
        pt1_idx, pt2_idx = connection
        if (pt1_idx < len(keypoints) and pt2_idx < len(keypoints) and
            keypoints[pt1_idx][2] > confidence_threshold and 
            keypoints[pt2_idx][2] > confidence_threshold):
            
            pt1 = (int(keypoints[pt1_idx][0]), int(keypoints[pt1_idx][1]))
            pt2 = (int(keypoints[pt2_idx][0]), int(keypoints[pt2_idx][1]))
            cv2.line(image, pt1, pt2, (0, 255, 0), 2)
    
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
            
            cv2.circle(image, (int(x), int(y)), 4, color, -1)
            
            # Add keypoint label
            if i < len(keypoint_names):
                cv2.putText(image, f"{keypoint_names[i][:3]}", 
                           (int(x) + 5, int(y) - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    # Add confidence info
    avg_confidence = np.mean([kp[2] for kp in keypoints if kp[2] > confidence_threshold])
    cv2.putText(image, f"Avg Conf: {avg_confidence:.3f}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Save visualization
    cv2.imwrite(output_path, image)
    print(f"Saved visualization to: {output_path}")
    
    # Print keypoint info
    print(f"\nDetected keypoints (confidence > {confidence_threshold}):")
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > confidence_threshold and i < len(keypoint_names):
            print(f"{keypoint_names[i]:>12}: ({x:6.1f}, {y:6.1f}) conf: {conf:.3f}")

if __name__ == "__main__":
    print("ViTPose CoreML Inference")
    print("=" * 30)
    
    # Run inference
    results = run_inference()
    
    if results is not None:
        print("\nInference completed successfully!")
        
        # Visualize results on both test images
        visualize_keypoints("../data/test_frame_1.jpg", results, "test_frame_1_keypoints.jpg")
        
        # Test with second image
        print("\n" + "="*50)
        print("Testing with second image...")
        print("="*50)
        
        # Load CoreML model again for second test
        model_2 = ct.models.MLModel("ViTPose_PyTorch.mlpackage")
        spec_2 = model_2._spec
        input_name_2 = spec_2.description.input[0].name
        output_name_2 = spec_2.description.output[0].name
        
        # Load and preprocess second image
        image_path_2 = "../data/test_frame_2.jpg"
        input_data_2 = preprocess_image(image_path_2)
        
        # Run inference on second image
        prediction_2 = model_2.predict({input_name_2: input_data_2})
        results_2 = prediction_2[output_name_2]
        
        # Visualize second image
        visualize_keypoints("../data/test_frame_2.jpg", {output_name_2: results_2}, "test_frame_2_keypoints.jpg")
    else:
        print("Inference failed.")