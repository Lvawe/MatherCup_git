import torch
print(torch.version.cuda)
if torch.cuda.is_available():
    print("GPU可用")
else:
    print("GPU不可用")
device_name = torch.cuda.get_device_name(torch.cuda.current_device())
print("当前使用的GPU设备：", device_name)