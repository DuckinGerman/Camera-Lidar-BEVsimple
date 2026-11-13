# import torch

# print("PyTorch:", torch.__version__)
# print("CUDA available:", torch.cuda.is_available())
# print("CUDA version:", torch.version.cuda)

# if torch.cuda.is_available():
#     print("GPU:", torch.cuda.get_device_name(0))
#     print("VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))

from nuscenes.nuscenes import NuScenes

nusc = NuScenes(
    version="v1.0-mini",
    dataroot=r"D:\bevtrain\data\nuscenes",
    verbose=True
)

print("Scenes:", len(nusc.scene))
print("Samples:", len(nusc.sample))
print("Annotations:", len(nusc.sample_annotation))




    