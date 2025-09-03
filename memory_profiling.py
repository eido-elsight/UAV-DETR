import warnings
warnings.filterwarnings('ignore')
import torch
import psutil
import os
from ultralytics import RTDETR

def get_gpu_memory_usage():
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0

def get_ram_usage():
    """Get current RAM usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2

def profile_model_memory():
    """Profile memory usage during model loading and inference"""
    
    print("🔍 UAV-DETR Memory Usage Profiling")
    print("=" * 50)
    
    # Baseline memory usage
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear GPU cache
        print(f"🔧 CUDA Device: {torch.cuda.get_device_name()}")
        print(f"🔧 Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    baseline_gpu = get_gpu_memory_usage()
    baseline_ram = get_ram_usage()
    
    print(f"\n📊 Baseline Memory Usage:")
    print(f"   GPU Memory: {baseline_gpu:.1f} MB")
    print(f"   RAM Usage: {baseline_ram:.1f} MB")
    
    # Load model and measure memory increase
    print(f"\n🚀 Loading UAV-DETR model...")
    model = RTDETR('runs/train/exp9/weights/best.pt')
    
    # Force model to GPU if available
    if torch.cuda.is_available():
        model.model = model.model.cuda()
        torch.cuda.synchronize()  # Wait for GPU operations to complete
    
    model_loaded_gpu = get_gpu_memory_usage()
    model_loaded_ram = get_ram_usage()
    
    gpu_increase = model_loaded_gpu - baseline_gpu
    ram_increase = model_loaded_ram - baseline_ram
    
    print(f"✅ Model Loaded Successfully!")
    print(f"📈 Memory After Loading:")
    print(f"   GPU Memory: {model_loaded_gpu:.1f} MB (+{gpu_increase:.1f} MB)")
    print(f"   RAM Usage: {model_loaded_ram:.1f} MB (+{ram_increase:.1f} MB)")
    
    # Test inference memory usage
    print(f"\n🔄 Running inference test...")
    test_image = 'test_images/overhead-cars.jpg'
    
    # Single image inference
    results = model.predict(source=test_image, imgsz=640, verbose=False, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    inference_gpu = get_gpu_memory_usage()
    inference_ram = get_ram_usage()
    
    gpu_inference_increase = inference_gpu - model_loaded_gpu
    ram_inference_increase = inference_ram - model_loaded_ram
    
    print(f"✅ Inference Complete!")
    print(f"📈 Memory During Inference:")
    print(f"   GPU Memory: {inference_gpu:.1f} MB (+{gpu_inference_increase:.1f} MB from loaded)")
    print(f"   RAM Usage: {inference_ram:.1f} MB (+{ram_inference_increase:.1f} MB from loaded)")
    
    # Summary
    print(f"\n📋 Memory Usage Summary:")
    print(f"   Model Loading: {gpu_increase:.1f} MB GPU, {ram_increase:.1f} MB RAM")
    print(f"   Inference Overhead: {gpu_inference_increase:.1f} MB GPU, {ram_inference_increase:.1f} MB RAM") 
    print(f"   Total for Inference: {inference_gpu:.1f} MB GPU, {inference_ram:.1f} MB RAM")
    
    # Theoretical calculations
    total_params = 45_456_520
    theoretical_size = total_params * 4 / 1024**2  # 4 bytes per float32 parameter
    print(f"\n🧮 Theoretical Calculations:")
    print(f"   Model Parameters: {total_params:,}")
    print(f"   Theoretical Size (float32): {theoretical_size:.1f} MB")
    print(f"   Actual GPU Usage: {gpu_increase:.1f} MB")
    
    if gpu_increase > 0:
        print(f"   Memory Efficiency: {theoretical_size/gpu_increase*100:.1f}% of theoretical minimum")
    else:
        print(f"   Note: Model not loaded to GPU during loading phase")

if __name__ == "__main__":
    profile_model_memory()
