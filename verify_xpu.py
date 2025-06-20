import torch
import intel_extension_for_pytorch as ipex

print("torch version:", torch.__version__)
print("ipex version:", ipex.__version__)
print("XPU available:", torch.xpu.is_available())
print("Device count:", torch.xpu.device_count())
