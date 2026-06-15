'''
Go to this link: https://www.kaggle.com/datasets/mitkir/shapenet?resource=download

Download the shapenetcore_partanno_segmentation_benchmark_v0_normal.zip
(19 directories, ...)

Download to /path/to/ShapeNet/raw/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip

Then:
# cd /path/to/ShapeNet/raw
# unzip shapenetcore_partanno_segmentation_benchmark_v0_normal.zip
# mv shapenetcore_partanno_segmentation_benchmark_v0_normal/* .
# rmdir shapenetcore_partanno_segmentation_benchmark_v0_normal

Verify the structure:
ShapeNet/
├── raw/
│   ├── 02691156/          (and other category folders)
│   ├── train_test_split/  (contains JSON files)
│   └── ... (other files)
└── processed/             (this will be created by PyG)
'''

from torch_geometric.datasets import ShapeNet
import torch_geometric.transforms as T
import torch

root_path = '/path/to/ShapeNet'
categories = ['Airplane', 'Motorbike', 'Car']

############################
## Firstime preprocessing ##
############################

shapenet = ShapeNet(
    root=root_path, 
    categories=categories,
    pre_transform=T.FixedPoints(2048)
)

'''
Or can use:

shapenet = ShapeNet(
    root=root_path, 
    categories=['Airplane', 'Motorbike', 'Car'],
    pre_transform=None, # Remove the transform here
    force_reload=True    # This overwrites the old 'processed' files
)
'''

print(f"Successfully loaded {len(shapenet)} shapes.")

#########################
## Get smaller dataset ##
#########################

indices = []
counts = {0: 0, 1: 0, 2: 0}
target_count = 10

for i in range(len(shapenet)):
    label = shapenet[i].category.item()
    if counts[label] < target_count:
        indices.append(i)
        counts[label] += 1
    
    # Stop if we found 10 for all 3 categories
    if all(c >= target_count for c in counts.values()):
        break

# 3. Create the subset
dataset = shapnet[torch.tensor(indices)]

print(f"Total shapes in subset: {len(dataset)}")
# Verify counts
for cat_idx, name in enumerate(categories):
    print(f"{name}: {len([d for d in dataset if d.category == cat_idx])} shapes")