import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from inference_dreds import Inferencer
import matplotlib.pyplot as plt
import os
import time


def show_images_and_depths(rgb, input_depth, pred_depth, gt_depth):
    titles = [
        "RGB", "Ground Truth Depth",
        "Simulated Depth (Input)", "Predicted Depth"
    ]
    images = [rgb / 255.0, gt_depth, input_depth, pred_depth]

    plt.figure(figsize=(16, 4))  # Wider figure for horizontal layout
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 4, i + 1)  # 1 row, 4 columns
        if i == 0:
            plt.imshow(img)
        else:
            plt.imshow(img, cmap='plasma', vmin=0.1, vmax=2)
            plt.colorbar(fraction=0.046, pad=0.04)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def show_images_and_depths_without_bar(rgb, input_depth, pred_depth, gt_depth):
    titles = [
        "RGB", "GT Depth",
        "Input Depth", "SwinDRNet"
    ]
    images = [rgb / 255.0, gt_depth, input_depth, pred_depth]

    plt.figure(figsize=(16, 4))  # Wider figure for horizontal layout
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 4, i + 1)  # 1 row, 4 columns
        if i == 0:
            plt.imshow(img)
        else:
            plt.imshow(img, cmap='plasma', vmin=0.1, vmax=2)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def show_images_depths_and_error(rgb, input_depth, pred_depth, gt_depth):
    error_map = np.abs(pred_depth - gt_depth)

    titles = [
        "RGB", "GT Depth",
        "Input Depth", "SwinDRNet", "Error Map"
    ]
    images = [rgb / 255.0, gt_depth, input_depth, pred_depth, error_map]

    plt.figure(figsize=(20, 4))  # Wider to fit 5 images
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 5, i + 1)
        if i == 0:
            plt.imshow(img)
        elif i == 4:
            plt.imshow(img, cmap='inferno')  # Error map: use 'inferno' for better visibility
            plt.colorbar(fraction=0.046, pad=0.04)
        else:
            plt.imshow(img, cmap='plasma', vmin=0.1, vmax=2)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def show_images_with_predictions_and_masked_errors(rgb, input_depth, pred_depth, gt_depth, mask):
    # Compute masked predictions and errors
    masked_pred = np.zeros_like(pred_depth)
    masked_pred[mask] = pred_depth[mask]

    masked_error_input = np.zeros_like(input_depth)
    masked_error_input[mask] = np.abs(input_depth[mask] - gt_depth[mask])

    titles = [
        "RGB", "GT Depth",
        "Input Depth", "SwinDRNet (Full)",
        "SwinDRNet (Masked)", "Input Error (Masked)"
    ]
    images = [
        rgb / 255.0, gt_depth, input_depth,
        pred_depth, masked_pred, masked_error_input
    ]

    plt.figure(figsize=(24, 4))  # Adjust width for 6 images
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 6, i + 1)
        if i == 0:
            plt.imshow(img)
        elif i == 5:
            plt.imshow(img, cmap='inferno', vmin=0, vmax=0.1)  # Error map
            plt.colorbar(fraction=0.046, pad=0.04)
        else:
            plt.imshow(img, cmap='plasma', vmin=0.1, vmax=2)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()



def showandsave_sim(rgb, input_depth, pred_depth, gt_depth, mask, save_dir="saved_outputs", prefix="scene"):
    os.makedirs(save_dir, exist_ok=True)

    # Compute error maps (masked)
    masked_error_pred = np.zeros_like(pred_depth)
    masked_error_pred[mask] = np.abs(pred_depth[mask] - gt_depth[mask])

    masked_error_input = np.zeros_like(input_depth)
    masked_error_input[mask] = np.abs(input_depth[mask] - gt_depth[mask])

    # Prepare images and titles
    titles = [
        "rgb", "gt_depth",
        "input_depth", "swin_full",
        "swin_error_masked", "input_error_masked"
    ]
    images = [
        rgb / 255.0, gt_depth, input_depth,
        pred_depth, masked_error_pred, masked_error_input
    ]

    # Save each image
    for title, img in zip(titles, images):
        save_path = os.path.join(save_dir, f"{prefix}_{title}.png")
        if title == "rgb":
            Image.fromarray((img * 255).astype(np.uint8)).save(save_path)
        else:
            plt.imsave(save_path, img, cmap='plasma' if 'depth' in title or 'swin_full' in title else 'inferno', vmin=0 if 'error' in title else 0.1, vmax=0.1 if 'error' in title else 2)

    # Plot as usual
    plt.figure(figsize=(24, 4))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 6, i + 1)
        if i == 0:
            plt.imshow(img)
        elif "error" in title:
            plt.imshow(img, cmap='inferno', vmin=0, vmax=0.1)
            plt.colorbar(fraction=0.046, pad=0.04)
        else:
            plt.imshow(img, cmap='plasma', vmin=0.2, vmax=1.5)
        plt.title(title.replace("_", " ").title())
        plt.axis('off')
    plt.tight_layout()
    plt.show()



def save_real(rgb, input_depth, pred_depth, save_dir="outputs", prefix="sample"):
    os.makedirs(save_dir, exist_ok=True)

    # Save RGB
    rgb_path = os.path.join(save_dir, f"{prefix}_rgb.png")
    Image.fromarray((rgb / 255.0 * 255).astype(np.uint8)).save(rgb_path)

    # Save input depth (visualized)
    input_vis_path = os.path.join(save_dir, f"{prefix}_input_depth.png")
    plt.imsave(input_vis_path, input_depth, cmap="plasma", vmin=0.2, vmax=0.8)

    # Save predicted depth (visualized)
    pred_vis_path = os.path.join(save_dir, f"{prefix}_pred_depth.png")
    plt.imsave(pred_vis_path, pred_depth, cmap="plasma", vmin=0.3, vmax=1)

    print(f"Saved RGB and depth images to: {save_dir}")


def draw_point_cloud(color, depth, camera_intrinsics, use_mask=False, use_inpainting=True, scale=1000.0, inpainting_radius=5, fault_depth_limit=0.2, epsilon=0.01):
    d = depth.copy()
    c = color.copy() / 255.0

    if use_inpainting:
        fault_mask = (d < fault_depth_limit * scale)
        d[fault_mask] = 0
        inpainting_mask = (np.abs(d) < epsilon * scale).astype(np.uint8)  
        d = cv2.inpaint(d, inpainting_mask, inpainting_radius, cv2.INPAINT_NS)

    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]

    xmap, ymap = np.arange(d.shape[1]), np.arange(d.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)

    points_z = d / scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z
    points = np.stack([points_x, points_y, points_z], axis=-1)

    if use_mask:
        mask = (points_z > 0)
        points = points[mask]
        c = c[mask]
    else:
        points = points.reshape((-1, 3))
        c = c.reshape((-1, 3))
        
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(c)
    return cloud

# Inference
inferencer = Inferencer()
rgb_real = np.array(Image.open('infer/azure_kinect_color0_image_0.png'), dtype = np.float32)
depth_real = np.array(Image.open('infer/azure_kinect_depth0_image_0.png'), dtype = np.float32)
depth_gt_real = np.array(Image.open('infer/depth1-gt.png'), dtype = np.float32)
scene = 'scene_000213'
img = '000018.png'
rgb_sim = np.array(Image.open(f'inference_samples_group/{scene}/bop_data/instrument/train_pbr/000000/rgb/{img}'), dtype=np.float32)
depth_sim = np.array(Image.open(f'inference_samples_group/{scene}/bop_data_new/instrument/train_pbr/000000/depth/{img}'), dtype=np.float32)
depth_gt_sim = np.array(Image.open(f'inference_samples_group/{scene}/bop_data/instrument/train_pbr/000000/depth/{img}'), dtype=np.float32)
mask= np.array(Image.open(f'inference_samples_group/{scene}/bop_data/instrument/train_pbr/000000/mask/{img}'), dtype=np.uint8)
mask = (mask > 0)
depth_sim = depth_sim / 1000
depth_gt_sim = depth_gt_sim / 1000

# Warm-up + Timed inference loop
num_warmup = 5
num_runs = 100

timings = []

for i in range(num_warmup + num_runs):
    start_time = time.time()
    res_sim, _ = inferencer.inference(rgb_sim, depth_sim, depth_coefficient=3, inpainting=True)
    end_time = time.time()

    # Skip warm-up rounds
    if i >= num_warmup:
        timings.append(end_time - start_time)

# Compute average
average_time = np.mean(timings)
std_time = np.std(timings)

print(f"Average inference time over {num_runs} runs (excluding {num_warmup} warm-up): {average_time:.4f} ± {std_time:.4f} seconds")

depth_real = depth_real / 1000
depth_gt_real = depth_gt_real / 1000

res_real, _ = inferencer.inference(rgb_real, depth_real, depth_coefficient=3, inpainting=True)

# Show RGB + Depths
# show_images_with_predictions_and_masked_errors(rgb, depth, res, depth_gt, mask)

# save the result 

# showandsave_sim(rgb_sim, depth_sim, res_sim, depth_gt_sim, mask,save_dir="res_dreds_exp1", prefix=f"sim_scene_{scene}_{img}")
# save_real(rgb_real, depth_real, res_real, save_dir="res_dreds_exp1", prefix=f"real_scene_{scene}_{img}")


# # cam_intrinsics = np.load('data/camera_intrinsics/1-camIntrinsics-D435.npy')
# cam_intrinsics = np.array([
#     [607.2491455078125,       0.0, 639.1669922024012],
#     [0.0,           607.1652573903423, 364.76153527267206],
#     [0.0,                         0.0,                 1.0]
# ])


# cloud = draw_point_cloud(rgb, depth, cam_intrinsics, scale = 1.0)
# # cloud_gt = draw_point_cloud(rgb, depth_gt, cam_intrinsics, scale = 1.0)

# frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
# sphere = o3d.geometry.TriangleMesh.create_sphere(0.002,20).translate([0,0,0.490])
# o3d.visualization.draw_geometries([cloud, frame, sphere])


