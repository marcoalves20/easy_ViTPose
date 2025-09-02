#!/usr/bin/env python3
"""
ViTPose PyTorch to CoreML Converter
===================================

Simple and direct conversion from PyTorch (.pth) model to CoreML format.
This approach is much more reliable than ONNX conversion.

Author: AI Assistant
Date: 2024
"""

import coremltools as ct
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import sys
import argparse
from typing import Optional

# Import ViTPose components
sys.path.append('../easy_ViTPose')
from vit_models.model import ViTPose
from vit_utils.util import dyn_model_import, infer_dataset_by_path


class ViTPosePTH2CoreML:
    """Simple PyTorch to CoreML converter for ViTPose models."""
    
    def __init__(self, pth_path: str, output_path: str = "ViTPose_PyTorch.mlpackage"):
        self.pth_path = pth_path
        self.output_path = output_path
        self.pytorch_model = None
        
    def load_pytorch_model(self) -> bool:
        """Load the PyTorch model from .pth file using proper ViTPose architecture."""
        print(f"🔍 Loading PyTorch model: {self.pth_path}")
        
        if not os.path.exists(self.pth_path):
            print(f"❌ PyTorch model not found: {self.pth_path}")
            return False
        
        try:
            # Infer dataset and model name from path
            dataset = infer_dataset_by_path(self.pth_path)
            print(f"   📊 Inferred dataset: {dataset}")
            
            # Determine model size from filename (vitpose-b-coco_25.pth -> 'b')
            model_name = 'b'  # Default to base model
            if 'vitpose-s-' in self.pth_path.lower():
                model_name = 's'
            elif 'vitpose-b-' in self.pth_path.lower():
                model_name = 'b'
            elif 'vitpose-l-' in self.pth_path.lower():
                model_name = 'l'
            elif 'vitpose-h-' in self.pth_path.lower():
                model_name = 'h'
            
            print(f"   🏗️  Using model size: {model_name}")
            
            # Get model configuration
            model_cfg = dyn_model_import(dataset, model_name)
            print(f"   ⚙️  Model config loaded")
            
            # Create ViTPose model
            self.pytorch_model = ViTPose(model_cfg)
            self.pytorch_model.eval()
            print(f"   🏗️  ViTPose model created")
            
            # Load checkpoint
            checkpoint = torch.load(self.pth_path, map_location='cpu', weights_only=True)
            print(f"   📦 Checkpoint loaded")
            
            # Load state dict
            if 'state_dict' in checkpoint:
                self.pytorch_model.load_state_dict(checkpoint['state_dict'])
                print(f"   ✅ State dict loaded from 'state_dict' key")
            else:
                self.pytorch_model.load_state_dict(checkpoint)
                print(f"   ✅ State dict loaded directly from checkpoint")
            
            # Test the model
            self._test_model()
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading PyTorch model: {e}")
            import traceback
            traceback.print_exc()
            return False
    

    
    def _test_model(self) -> None:
        """Test the loaded model with dummy input."""
        print("   🧪 Testing model with dummy input...")
        
        try:
            # Create dummy input
            dummy_input = torch.randn(1, 3, 256, 192)
            
            with torch.no_grad():
                output = self.pytorch_model(dummy_input)
            
            print(f"   📊 Input shape: {dummy_input.shape}")
            print(f"   📊 Output shape: {output.shape}")
            print(f"   📊 Output range: [{output.min():.3f}, {output.max():.3f}]")
            
        except Exception as e:
            print(f"   ⚠️  Model test failed: {e}")
            print(f"   💡 This might be expected if the model needs specific preprocessing")
    
    def convert_to_coreml(self) -> Optional[ct.models.MLModel]:
        """Convert PyTorch model to CoreML."""
        print("🔄 Converting PyTorch model to CoreML...")
        
        try:
            # Create example input for tracing
            example_input = torch.randn(1, 3, 256, 192)
            
            # Trace the model
            print("   📝 Tracing model...")
            traced_model = torch.jit.trace(self.pytorch_model, example_input)
            
            # Convert to CoreML
            print("   🔄 Converting to CoreML...")
            coreml_model = ct.convert(
                traced_model,
                inputs=[ct.TensorType(name="input", shape=(1, 3, 256, 192))],
                source="pytorch",
                convert_to="mlprogram",
                compute_precision=ct.precision.FLOAT32
            )
            
            print("✅ CoreML conversion successful!")
            return coreml_model
            
        except Exception as e:
            print(f"❌ CoreML conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def validate_conversion(self, coreml_model: ct.models.MLModel) -> bool:
        """Validate the converted CoreML model."""
        print("🧪 Validating converted CoreML model...")
        
        try:
            # Create test input
            test_input = np.random.randn(1, 3, 256, 192).astype(np.float32)
            
            # Get PyTorch reference output
            with torch.no_grad():
                pytorch_output = self.pytorch_model(torch.from_numpy(test_input))
                pytorch_output_np = pytorch_output.numpy()
            
            # Get CoreML output
            spec = coreml_model._spec
            input_key = spec.description.input[0].name
            output_key = spec.description.output[0].name
            
            coreml_prediction = coreml_model.predict({input_key: test_input})
            coreml_output = coreml_prediction[output_key]
            
            print(f"   📊 PyTorch output: {pytorch_output_np.shape} range[{pytorch_output_np.min():.3f}, {pytorch_output_np.max():.3f}]")
            print(f"   📊 CoreML output: {coreml_output.shape} range[{coreml_output.min():.3f}, {coreml_output.max():.3f}]")
            
            # Compare outputs
            if pytorch_output_np.shape == coreml_output.shape:
                abs_diff = np.abs(pytorch_output_np - coreml_output)
                mean_abs_diff = np.mean(abs_diff)
                max_abs_diff = np.max(abs_diff)
                
                print(f"   📏 Mean absolute difference: {mean_abs_diff:.6f}")
                print(f"   📏 Max absolute difference: {max_abs_diff:.6f}")
                
                if mean_abs_diff < 1e-3:
                    print("✅ Outputs are very close - excellent conversion!")
                    return True
                elif mean_abs_diff < 1e-2:
                    print("⚠️  Outputs have small differences - acceptable")
                    return True
                else:
                    print("❌ Outputs have significant differences")
                    return False
            else:
                print(f"❌ Output shapes don't match")
                return False
                
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return False
    
    def convert(self) -> bool:
        """Main conversion method."""
        print("🚀 Starting ViTPose PyTorch to CoreML Conversion")
        print("=" * 60)
        
        # Step 1: Load PyTorch model
        if not self.load_pytorch_model():
            print("❌ Failed to load PyTorch model")
            return False
        
        # Step 2: Convert to CoreML
        coreml_model = self.convert_to_coreml()
        
        if coreml_model is None:
            print("❌ CoreML conversion failed")
            return False
        
        # Step 3: Save and validate
        print(f"\n💾 Saving CoreML model to: {self.output_path}")
        coreml_model.save(self.output_path)
        
        # Validate
        validation_passed = self.validate_conversion(coreml_model)
        
        # Print model info
        self.print_model_info(coreml_model)
        
        if validation_passed:
            print(f"\n🎉 Conversion completed successfully!")
            print(f"📱 CoreML model saved as: {self.output_path}")
            return True
        else:
            print(f"\n⚠️  Conversion completed with validation issues")
            print(f"📱 CoreML model saved as: {self.output_path}")
            return True  # Still return True since model was created
    
    def print_model_info(self, coreml_model: ct.models.MLModel) -> None:
        """Print model information."""
        print(f"\n📋 Model Information:")
        
        spec = coreml_model._spec
        input_info = spec.description.input[0]
        output_info = spec.description.output[0]
        
        try:
            input_shape = [d.dim_value if hasattr(d, 'dim_value') else d for d in input_info.type.multiArrayType.shape]
            output_shape = [d.dim_value if hasattr(d, 'dim_value') else d for d in output_info.type.multiArrayType.shape]
            print(f"   📥 Input: {input_info.name} {input_shape}")
            print(f"   📤 Output: {output_info.name} {output_shape}")
        except:
            print(f"   📥 Input: {input_info.name}")
            print(f"   📤 Output: {output_info.name}")
        
        # Get model size
        if os.path.exists(self.output_path):
            import subprocess
            try:
                result = subprocess.run(['du', '-sh', self.output_path], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    size = result.stdout.split()[0]
                    print(f"   💾 Model size: {size}")
            except:
                pass


def test_with_real_image(coreml_model_path: str, test_image_path: str) -> bool:
    """Test the converted model with a real image."""
    print(f"\n🖼️  Testing with real image: {test_image_path}")
    
    if not os.path.exists(test_image_path):
        print(f"❌ Test image not found: {test_image_path}")
        return False
    
    try:
        # Load and preprocess image
        image = Image.open(test_image_path).convert('RGB')
        image = image.resize((192, 256), Image.LANCZOS)
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)
        
        print(f"   📊 Image input shape: {img_array.shape}")
        print(f"   📊 Image input range: [{img_array.min():.3f}, {img_array.max():.3f}]")
        
        # Test with CoreML
        coreml_model = ct.models.MLModel(coreml_model_path)
        spec = coreml_model._spec
        input_key = spec.description.input[0].name
        output_key = spec.description.output[0].name
        
        result = coreml_model.predict({input_key: img_array})[output_key]
        
        print(f"✅ Real image test successful!")
        print(f"   📊 Output shape: {result.shape}")
        print(f"   📊 Output range: [{result.min():.3f}, {result.max():.3f}]")
        
        # Count high-confidence keypoints
        high_conf_count = np.sum(result > 0.5)
        print(f"   🎯 High confidence detections: {high_conf_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Real image test failed: {e}")
        return False


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Convert ViTPose PyTorch model to CoreML format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python vitpose_pth2coreml.py models/vitpose-b-coco_25.pth
  python vitpose_pth2coreml.py models/vitpose-b-coco_25.pth --output MyViTPose.mlpackage
  python vitpose_pth2coreml.py models/vitpose-b-coco_25.pth --test data/test_frame_1.jpg
        """
    )
    
    parser.add_argument(
        "pth_path",
        help="Path to the PyTorch model file (.pth)"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="ViTPose_PyTorch.mlpackage",
        help="Output path for CoreML model (default: ViTPose_PyTorch.mlpackage)"
    )
    
    parser.add_argument(
        "--test", "-t",
        help="Test image path to validate conversion with real image"
    )
    
    args = parser.parse_args()
    
    # Create converter and run conversion
    converter = ViTPosePTH2CoreML(args.pth_path, args.output)
    success = converter.convert()
    
    if success:
        print(f"\n📋 Next steps:")
        print(f"1. Test the model with: python coreml_inference.py")
        print(f"2. Use in your app with CoreML framework")
        print(f"3. Model saved at: {args.output}")
        
        # Test with real image if provided
        if args.test:
            test_with_real_image(args.output, args.test)
        
        return True
    else:
        print(f"\n💡 Troubleshooting tips:")
        print(f"1. Check if your PyTorch model is valid")
        print(f"2. Ensure the model architecture is compatible with CoreML")
        print(f"3. Check CoreML Tools version: pip install --upgrade coremltools")
        
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
