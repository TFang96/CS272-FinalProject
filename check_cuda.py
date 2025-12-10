import torch as th
import sys

def check_cuda_setup():
    """Checks for CUDA availability and provides detailed information."""
    print("--- PyTorch CUDA Setup Check ---")
    print(f"PyTorch Version: {th.__version__}")
    
    # Check for CUDA availability
    if th.cuda.is_available():
        print("\n✅ CUDA is **AVAILABLE**!")
        
        # Get the number of GPUs
        num_gpus = th.cuda.device_count()
        print(f"Total number of GPUs detected: {num_gpus}")
        
        # Get system-wide CUDA version (PyTorch was built with)
        # Note: This is the version PyTorch links against.
        print(f"PyTorch built with CUDA version: {th.version.cuda}")
        
        # Report details for each detected GPU
        for i in range(num_gpus):
            device_name = th.cuda.get_device_name(i)
            device_capability = th.cuda.get_device_capability(i)
            print(f"\n--- GPU {i} Details ---")
            print(f"Device Name: {device_name}")
            print(f"Compute Capability (Arch): {device_capability[0]}.{device_capability[1]}")
            
            # Test a small tensor on the GPU
            try:
                tensor_gpu = th.ones(2, 2).to(f'cuda:{i}')
                print(f"Successfully allocated a 2x2 tensor on device {i}.")
            except Exception as e:
                print(f"⚠️ Error allocating tensor on GPU {i}: {e}")
                
    else:
        print("\n❌ CUDA is **NOT AVAILABLE**!")
        print("This usually means:")
        print("1. NVIDIA drivers are not installed or configured correctly.")
        print("2. The installed PyTorch version was not built with CUDA support.")
        print("3. PyTorch cannot detect your GPU.")
        
    print("\n--------------------------------")

if __name__ == "__main__":
    # Ensure PyTorch is imported before running the check
    if 'torch' in sys.modules:
        check_cuda_setup()
    else:
        print("Error: PyTorch is not installed or available.")