import os
import site
import sys

def setup_gpu_env() -> None:
    """
    Sets the required LD_LIBRARY_PATH for TensorFlow to recognize the GPU in WSL2.
    
    Why this is needed:
    When installing TensorFlow via pip in WSL2, the NVIDIA CUDA libraries 
    (like cuBLAS, cuDNN) are installed into the Python 'site-packages/nvidia' directory. 
    Linux does not look for C-libraries in Python directories by default. 
    This function dynamically locates the active Python environment and appends 
    the NVIDIA library paths to the system's LD_LIBRARY_PATH before TensorFlow initializes.
    """
    # 1. Dynamically find the active site-packages directories (works for venv, conda, user installs)
    site_packages = site.getsitepackages() 
    if hasattr(site, 'getusersitepackages'):
        site_packages.append(site.getusersitepackages())

    # 2. Locate the 'nvidia' package folder
    nvidia_base = None
    for sp in site_packages:
        potential_path = os.path.join(sp, "nvidia")
        if os.path.exists(potential_path):
            nvidia_base = potential_path
            break
            
    if not nvidia_base:
        print("⚠️ Warning: 'nvidia' directory not found in site-packages. GPU might not be detected if relying on pip-installed CUDA.")
    else:
        # 3. Construct paths for all required NVIDIA sub-libraries
        libs = [
            "cusolver", "cusparse", "cudnn", "nccl", "cuda_nvrtc", "nvtx",
            "cuda_runtime", "cublas", "nvjitlink", "cufft", "cuda_cupti", "curand"
        ]
        nvidia_paths = ":".join([f"{nvidia_base}/{lib}/lib" for lib in libs])
        
        # 4. Append paths to the environment safely
        current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        wsl_lib = "/usr/lib/wsl/lib"
        
        # Check if we are actually running inside WSL2
        if os.path.exists(wsl_lib):
            os.environ['LD_LIBRARY_PATH'] = f"{nvidia_paths}:{wsl_lib}:{current_ld_path}"
        else:
            # Native Linux (e.g. your HPC cluster)
            os.environ['LD_LIBRARY_PATH'] = f"{nvidia_paths}:{current_ld_path}"
        
    # =========================================================================
    # CRITICAL: TensorFlow must be imported AFTER os.environ is modified!
    # =========================================================================
    import tensorflow as tf
    
    # 5. Verify GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            details = tf.config.experimental.get_device_details(gpus[0])
            gpu_name = details.get('device_name', 'Unknown GPU')
        except Exception:
            gpu_name = "GPU (Details unavailable)"
        print(f"✅ SUCCESS: TensorFlow is using {gpu_name}")
    else:
        print("⚠️ No GPU found by TensorFlow. Falling back to CPU.")