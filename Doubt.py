import torch
print(torch.__version__)

print(torch.backends.quantized.supported_engines)
# 2.4.0+cpu
# ['qnnpack', 'none', 'onednn', 'x86', 'fbgemm']
