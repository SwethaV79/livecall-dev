import torch

print('torch.cuda.is_available():', torch.cuda.is_available())
print('torch.cuda.device_count():', torch.cuda.device_count())
if torch.cuda.is_available():
    print('torch.cuda.get_device_name(0):', torch.cuda.get_device_name(0))
else:
    print('No GPU detected by PyTorch')
