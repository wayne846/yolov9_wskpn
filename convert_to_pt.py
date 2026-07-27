"""
將輸入的exr轉換成tensor格式
"""

import os
import pyexr
import numpy as np
import torch
from tqdm import tqdm

def Padding(img, w):
    return np.pad(img, ((w, w), (w, w), (0, 0)))

def convert():
    folder_name = "dataset"
    # 將這裡替換成你實際有的場景名稱
    scene_names = ["arcade", "bistro", "living_room", "sibenik", "sponza", "subway"]
    img_num_per_scene = 79

    for scene in scene_names:
        save_dir = os.path.join(folder_name, scene, "tensors")
        os.makedirs(save_dir, exist_ok=True)
        print(f"Processing scene: {scene}")

        for i in tqdm(range(img_num_per_scene)):
            f_irradiance = os.path.join(folder_name, scene, "acc_colors", f"color{i}.exr")
            f_albedo = os.path.join(folder_name, scene, "inputs", f"albedo{i}.exr")
            f_normal = os.path.join(folder_name, scene, "inputs", f"shading_normal{i}.exr")
            f_depth = os.path.join(folder_name, scene, "depth", f"depth{i}.exr")
            f_ref = os.path.join(folder_name, scene, "inputs", f"reference{i}.exr")
            f_roughness = os.path.join(folder_name, scene, "inputs", f"roughness{i}.exr")

            if not os.path.exists(f_ref): 
                continue # 如果檔案不存在就跳過

            # 讀取並強制轉為 3 通道 (RGB)
            irradiance_img = pyexr.read(f_irradiance)[:, :, :3]
            albedo_img = pyexr.read(f_albedo)[:, :, :3]
            reference_img = pyexr.read(f_ref)[:, :, :3]
            normal_img = pyexr.read(f_normal)[:, :, :3]
            depth_img = pyexr.read(f_depth)[:, :, 0:1]
            roughness_img = pyexr.read(f_roughness)[:, :, 3:4]

            # 打包成 Dict 並存檔為 PyTorch Tensor
            tensor_dict = {
                'irradiance': torch.tensor(irradiance_img, dtype=torch.float16),
                'albedo': torch.tensor(albedo_img, dtype=torch.float16),
                'normal': torch.tensor(normal_img, dtype=torch.float16),
                'depth': torch.tensor(depth_img, dtype=torch.float16),
                'roughness': torch.tensor(roughness_img, dtype=torch.float16),
                'targets': torch.tensor(reference_img, dtype=torch.float16)
            }
            torch.save(tensor_dict, os.path.join(save_dir, f"frame_{i}.pt"))

if __name__ == '__main__':
    convert()