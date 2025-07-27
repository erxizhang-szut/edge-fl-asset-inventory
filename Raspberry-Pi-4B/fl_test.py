# fl_test.py
import torch
import time

def rpi_hardware_info():
    """Get Raspberry Pi hardware information"""
    info = {
        "cpu_temp": float(open('/sys/class/thermal/thermal_zone0/temp').read())/1000,
        "model": open('/proc/device-tree/model').read().strip('\x00')
    }
    return info

print("Raspberry Pi 4B Edge Node Ready")
print(f"Device Info: {rpi_hardware_info()}")

# Test PyTorch installation
x = torch.rand(5, 3)
print(f"PyTorch Tensor: {x}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")  # Should be False on RPi

# Performance benchmark
start = time.time()
a = torch.rand(1000, 1000)
b = torch.rand(1000, 1000)
c = torch.mm(a, b)
duration = time.time() - start
print(f"Matrix Multiplication (1000x1000): {duration:.4f} seconds")
