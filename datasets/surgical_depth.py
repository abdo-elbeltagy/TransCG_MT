import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from utils.data_preparation import process_data


class POPSurgicalDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, depth_norm=1.0, **kwargs):
        """
        root_dir: Path to your main dataset directory containing all scenes.
        split: 'train' or 'test' (if your dataset has explicit splits).
        transform: torchvision transforms (optional) for augmentation.
        depth_norm: normalization factor for depth maps.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.depth_norm = depth_norm
        self.image_size = kwargs.get('image_size', (1280, 720))
        self.sample_info = []
        self.depth_min = kwargs.get('depth_min', 0.3)
        self.depth_max = kwargs.get('depth_max', 1.5)
        self.with_original = kwargs.get('with_original', False)
        scenes = [d for d in os.listdir(self.root_dir) if d.startswith('scene_')]
        scenes.sort()

        for scene in scenes:
            bop_path = os.path.join(self.root_dir, scene, 'bop_data', 'instrument', 'train_pbr', '000000')
            bop_new_path = os.path.join(self.root_dir, scene, 'bop_data_new', 'instrument', 'train_pbr', '000000')

            rgb_folder = os.path.join(bop_path, 'rgb')
            depth_gt_folder = os.path.join(bop_path, 'depth')
            mask_folder = os.path.join(bop_path, 'mask')
            depth_simulated_folder = os.path.join(bop_new_path, 'depth')
            camera_json_path = os.path.join(bop_path, 'scene_camera.json')

            # Load camera intrinsics once per scene
            with open(camera_json_path, 'r') as f:
                camera_intrinsics = json.load(f)

            image_files = sorted(os.listdir(rgb_folder))

            for img_file in image_files:
                img_id = img_file.split('.')[0]
                rgb_path = os.path.join(rgb_folder, img_file)
                depth_gt_path = os.path.join(depth_gt_folder, img_file)
                depth_sim_path = os.path.join(depth_simulated_folder, img_file)
                mask_path = os.path.join(mask_folder, img_file)

                self.sample_info.append({
                    'rgb': rgb_path,
                    'depth_gt': depth_gt_path,
                    'depth_sim': depth_sim_path,
                    'mask': mask_path,
                    'intrinsics': camera_intrinsics[str(int(img_id))]['cam_K']
                })

    def __len__(self):
        return len(self.sample_info)

    def __getitem__(self, idx):
        sample = self.sample_info[idx]

        rgb = Image.open(sample['rgb']).convert('RGB')
        depth_gt = Image.open(sample['depth_gt'])
        depth_sim = Image.open(sample['depth_sim'])
        mask = Image.open(sample['mask'])

        rgb = np.array(rgb, dtype=np.float32)
        # #print(np.max(depth_gt))
        depth_gt = np.array(depth_gt, dtype=np.float32)
        depth_sim = np.array(depth_sim, dtype=np.float32)
        mask = np.array(mask, dtype=np.uint8)
        intrinsics = np.array(sample['intrinsics'], dtype=np.float32).reshape(3, 3)

        return process_data(
            rgb, depth_sim, depth_gt, mask, intrinsics,
            scene_type = "cluttered",
            camera_type = 1,
            split = self.split, 
            image_size = self.image_size, 
            depth_min = self.depth_min, 
            depth_max = self.depth_max, 
            depth_norm = self.depth_norm, 
            use_aug = False, 
            with_original = self.with_original
            )

        #return data
