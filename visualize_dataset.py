import matplotlib.pyplot as plt
from datasets.surgical_depth import POPSurgicalDataset  # adjust if path is different

# Create the dataset
dataset = POPSurgicalDataset(
    root_dir='/home/beltagy/Desktop/master_thesis/BlenderProc/examples/datasets/OP_room/output',
    split='train',
    depth_norm=1.0,
    image_size=(320, 240),
    with_original=True
)

# Pick an index to visualize
sample = dataset[9]
print(sample)
# Extract tensors
rgb = sample['rgb'].permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]
depth_sim = sample['depth'].squeeze().numpy()
depth_gt = sample['depth_gt'].squeeze().numpy()
mask = sample['depth_gt_mask'].squeeze().numpy()

# Plot everything
plt.figure(figsize=(12, 4))

plt.subplot(1, 4, 1)
plt.imshow(rgb)
plt.title('RGB')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(depth_sim, cmap='inferno')
plt.title('Simulated Depth')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(depth_gt, cmap='inferno')
plt.title('Ground Truth Depth')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(mask, cmap='gray')
plt.title('Mask')
plt.axis('off')

plt.tight_layout()
plt.show()
