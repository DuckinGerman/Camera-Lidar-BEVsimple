
from collections import Counter
from nuscenes.nuscenes import NuScenes

from datasets.nuscenes_dataset import find_available_samples

DATAROOT = r"D:\bevtrain\data\nuscenes"

nusc = NuScenes(
    version="v1.0-trainval",
    dataroot=DATAROOT,
    verbose=False
)

available = find_available_samples(nusc)

scenes = set(s["scene_token"] for s in available)

print("Total metadata samples:", len(nusc.sample))
print("Available complete samples:", len(available))
print("Available scenes:", len(scenes))

categories = Counter()

for sample in available:
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        categories[ann["category_name"]] += 1

print("\nTop categories:")
for name, count in categories.most_common(15):
    print(f"{name}: {count}")

if not available:
    raise RuntimeError(
        "No complete samples found. Check extraction paths."
    )
