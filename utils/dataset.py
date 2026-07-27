import numpy as np
import os
from tqdm import tqdm
import pyexr
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

def robust_normalize(img):
    p_low = np.percentile(img, 1)
    p_high = np.percentile(img, 99.9999)
    img_clipped = np.clip(img, p_low, p_high)    
    return img_clipped

def Normalize(img):
    img = img.astype(np.float32)
    normalized = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-6) * (1.0 - 0.0)
    return normalized

def Padding(img, w):
    return np.pad(img, ((w, w), (w, w), (0, 0)))

class DataBase:
    def __init__(self, crop_size=128):
        folder_name = os.path.join("dataset")
        scene_names = ["bistro"]
            
        img_num_per_scene = 40
        tensor_file_names = [os.path.join(folder_name, scene_name, "tensors", "frame_"+str(i)+".pt") for scene_name in scene_names for i in range(img_num_per_scene)]

        self.train_inputs, self.train_targets = [], []
        self.test_inputs, self.test_targets = [], []
        
        for i in tqdm(range(len(tensor_file_names))):
            # 讀取 .pt 檔案
            data = torch.load(tensor_file_names[i])
            
            # 取出並轉為 float32 (因為學長的 numpy 函式需要進行浮點數運算)
            # 1. Irradiance 套用 robust_normalize
            irradiance_img = robust_normalize(data['irradiance'].numpy().astype(np.float32))
            
            # 2. Albedo 套用 Normalize
            albedo_img = Normalize(data['albedo'].numpy().astype(np.float32))
            
            # 3. Normal 與 Depth 恢復原本的正規化計算
            normal_img = data['normal'].numpy().astype(np.float32) * 0.5 + 0.5
            depth_img = data['depth'].numpy().astype(np.float32)
            depth_img = (depth_img - np.min(depth_img)) / (np.max(depth_img) - np.min(depth_img) + 1e-4)
            
            # 4. Roughness 套用 Min-Max 正規化
            roughness_img = data['roughness'].numpy().astype(np.float32)
            roughness_img = (roughness_img - np.min(roughness_img)) / (np.max(roughness_img) - np.min(roughness_img) + 1e-4)
            
            targets_raw = data['targets'].numpy().astype(np.float32)

            # 依照學長定義的順序進行串接 (3 + 3 + 1 + 3 + 1 = 11 通道)
            inputs_raw = np.concatenate((irradiance_img,
                                         albedo_img,
                                         roughness_img,
                                         normal_img,
                                         depth_img), axis=2)

            # 處理 NaN
            inputs = np.nan_to_num(inputs_raw, nan=0.0, posinf=1.0, neginf=0.0)
            targets = np.nan_to_num(targets_raw, nan=0.0, posinf=1.0, neginf=0.0)

            if i <(img_num_per_scene * 0.8) - 1:
                inputs_padded = Padding(inputs, crop_size)
                targets_padded = Padding(targets, crop_size)

                self.train_inputs.append(inputs_padded)
                self.train_targets.append(targets_padded)
            else:
                self.test_inputs.append(inputs)
                self.test_targets.append(targets)
                
        # 讀取第一張圖來設定影像邊界
        H, W, _ = self.test_targets[0].shape
        self.img_h, self.img_w = H - crop_size, W - crop_size



class BMFRFullResAlDataset(Dataset):
    def __init__(self, database, use_train=False, use_val=False, use_test=False, train_crops_every_frame=77, val_crops_every_frame=20, crop_size=128): # BMFR
        self.database = database
        self.use_train = use_train
        self.use_val = use_val
        self.use_test = use_test
        self.train_crops_every_frame = train_crops_every_frame
        self.val_crops_every_frame = val_crops_every_frame
        self.crop_size = crop_size

        def rotate90(inputs):
            inputs = torch.rot90(inputs, 1, (1, 2))
            return inputs
        def rotate270(inputs):
            inputs = torch.rot90(inputs, -1, (1, 2))
            return inputs
        self.transforms = [TF.hflip, TF.vflip, rotate90, rotate270]
        
            
    def _apply_transform(self, input_img, target_img):
        if self.use_train or self.use_val:
            # Random crop and convert ndarray to tensor
            i, j = np.random.randint(self.database.img_h - self.crop_size), np.random.randint(self.database.img_w-self.crop_size)
            input_crop = TF.to_tensor(input_img[i:i+self.crop_size, j:j+self.crop_size].astype(np.float32))
            target_crop = TF.to_tensor(target_img[i:i+self.crop_size, j:j+self.crop_size].astype(np.float32))
            
            if np.random.rand() > 0.5:
                transform = np.random.choice(self.transforms)
                input_crop = transform(input_crop)
                target_crop = transform(target_crop)
        elif self.use_test:
            input_crop = TF.to_tensor(input_img.astype(np.float32))
            target_crop = TF.to_tensor(target_img.astype(np.float32))
            
        return input_crop, target_crop
        
    def __getitem__(self, idx):
        if self.use_test:
            frame_idx = idx
            inputs = self.database.test_inputs[frame_idx]
            targets = self.database.test_targets[frame_idx]
        elif self.use_train:
            frame_idx = idx // self.train_crops_every_frame
            inputs = self.database.train_inputs[frame_idx]
            targets = self.database.train_targets[frame_idx]
        elif self.use_val:
            frame_idx = idx // self.val_crops_every_frame
            inputs = self.database.train_inputs[frame_idx]
            targets = self.database.train_targets[frame_idx]
            
        return self._apply_transform(inputs, targets)
    
    def __len__(self):
        if self.use_train:
            return len(self.database.train_targets) * self.train_crops_every_frame
        elif self.use_val:
            return len(self.database.train_targets) * self.val_crops_every_frame
        elif self.use_test:
            return len(self.database.test_targets)