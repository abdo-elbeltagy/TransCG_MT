import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from inference_dreds import Inferencer
import matplotlib.pyplot as plt

def show_images_and_depths(rgb, input_depth, pred_depth, gt_depth):
    titles = [
        "RGB", "Ground Truth Depth",
        "Simulated Depth (Input)", "Predicted Depth"
    ]
    images = [rgb / 255.0, gt_depth, input_depth, pred_depth]

    plt.figure(figsize=(12, 8))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(2, 2, i + 1)
        if i == 0:
            plt.imshow(img)
        else:
            plt.imshow(img, cmap='plasma', vmin=0.1, vmax=2)
            plt.colorbar(fraction=0.046, pad=0.04)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

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
rgb = np.array(Image.open('infer/azure_kinect_color0_image_0.png'), dtype = np.float32)
depth = np.array(Image.open('infer/azure_kinect_depth0_image_0.png'), dtype = np.float32)
depth_gt = np.array(Image.open('infer/depth1-gt.png'), dtype = np.float32)
# rgb = np.array(Image.open('inference_samples_transcg/scene_000213/bop_data/instrument/train_pbr/000000/rgb/000003.png'), dtype=np.float32)
# depth = np.array(Image.open('inference_samples_transcg/scene_000213/bop_data_new/instrument/train_pbr/000000/depth/000003.png'), dtype=np.float32)
# depth_gt = np.array(Image.open('inference_samples_transcg/scene_000213/bop_data/instrument/train_pbr/000000/depth/000003.png'), dtype=np.float32)

depth = depth / 1000
depth_gt = depth_gt / 1000

res, _ = inferencer.inference(rgb, depth, depth_coefficient=3, inpainting=True)

# Show RGB + Depths
show_images_and_depths(rgb, depth, res, depth_gt)

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


