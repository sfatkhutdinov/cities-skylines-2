"""
Test script for the visual metrics estimator.
Verifies that the module can be imported and used with GPU acceleration.
"""

import sys
import os
import time
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import the components directly
try:
    # Import directly from module paths to avoid circular dependencies
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
    from config.hardware_config import HardwareConfig
    
    # Import the visual metrics class too
    from environment.visual_metrics import VisualMetricsEstimator
    print("Successfully imported modules")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def test_visual_metrics():
    """Test that the visual metrics estimator works with GPU acceleration."""
    # Create hardware config
    config = HardwareConfig()
    
    # Print hardware info
    print(f"\nRunning on: {config.device}")
    print(f"Using mixed precision: {config.mixed_precision}")
    
    # Create estimator
    metrics_estimator = VisualMetricsEstimator(config)
    
    # Verify model is on correct device
    model_device = next(metrics_estimator.feature_extractor.parameters()).device
    print(f"Model device: {model_device}")
    
    # Create a dummy frame (RGB image as tensor)
    height, width = 240, 320
    dummy_frame = torch.rand(3, height, width, device=config.get_device(), dtype=config.get_dtype())
    print(f"Dummy frame shape: {dummy_frame.shape}")
    print(f"Dummy frame device: {dummy_frame.device}")
    print(f"Dummy frame dtype: {dummy_frame.dtype}")
    
    # Test feature extraction
    start_time = time.time()
    features = metrics_estimator._extract_features(dummy_frame)
    end_time = time.time()
    
    print(f"Feature extraction time: {(end_time - start_time)*1000:.2f} ms")
    print(f"Features shape: {features.shape}")
    print(f"Features device: {features.device}")
    
    # Test population estimation
    start_time = time.time()
    population, metrics = metrics_estimator.estimate_population(dummy_frame)
    end_time = time.time()
    
    print(f"Population estimation time: {(end_time - start_time)*1000:.2f} ms")
    print(f"Estimated population: {population}")
    print(f"Metrics: {metrics}")
    
    print("\nGPU memory status:")
    if torch.cuda.is_available():
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.4f} GB")
        print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.4f} GB")
    
    print("\nTest completed successfully")

if __name__ == "__main__":
    test_visual_metrics() 