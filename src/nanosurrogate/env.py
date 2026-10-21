import os
import tensorflow as tf

def setup_gpu_env():
    """Sets WSL2 NVIDIA library paths before TensorFlow is imported."""
    base = "/home/jonasmensing/.local/lib/python3.10/site-packages/nvidia"
    libs = [
        "cusolver", "cusparse", "cudnn", "nccl", "cuda_nvrtc", "nvtx",
        "cuda_runtime", "cublas", "nvjitlink", "cufft", "cuda_cupti", "curand"
    ]
    nvidia_paths = ":".join([f"{base}/{lib}/lib" for lib in libs])
    wsl_lib = "/usr/lib/wsl/lib"
    os.environ['LD_LIBRARY_PATH'] = f"{nvidia_paths}:{wsl_lib}:" + os.environ.get('LD_LIBRARY_PATH', '')

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        details = tf.config.experimental.get_device_details(gpus[0])
        print(f"GPU Ready: {details.get('device_name', 'Unknown GPU')}")
    else:
        print("No GPU found. Running on CPU.")