import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import cv2
from typing import Optional, Callable, Dict
import random


def fixed_bilinear_resizing(wmap, size=(128, 128)):
    wmap_preprocessed = cv2.resize(wmap, size, interpolation=cv2.INTER_LINEAR)
    return wmap_preprocessed

def fixed_resizing(wmap, size=(128, 128)):
    wmap_preprocessed = cv2.resize(wmap, size, interpolation=cv2.INTER_NEAREST)
    return wmap_preprocessed

def felexible_resizing(wmap, size=(128, 128)):
    h, w = wmap.shape
    target_h, target_w = size

    # scale factor (aspect ratio)
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # nearest interpolation (To preserve categorical values)
    resized = cv2.resize(wmap, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # zero padding
    padded = np.zeros((target_h, target_w), dtype=wmap.dtype)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    padded[top:top + new_h, left:left + new_w] = resized

    return padded

def wafer_letterbox(wmap, new_shape=(128, 128), pad_value=0, auto=False, scaleFill=False, scaleup=True, stride=32):
    """
    YOLO-style letterbox resizing for wafer bin maps.
    """

    shape = wmap.shape[:2]  # (height, width)

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    target_h, target_w = new_shape

    # Scale ratio
    r = min(target_h / shape[0], target_w / shape[1])

    # Do not scale up if scaleup=False
    if not scaleup:
        r = min(r, 1.0)

    # New size before padding: (width, height)
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))

    # Padding
    dw = target_w - new_unpad[0]
    dh = target_h - new_unpad[1]

    if auto:
        dw = np.mod(dw, stride)
        dh = np.mod(dh, stride)

    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (target_w, target_h)

    dw /= 2
    dh /= 2

    # Resize
    if shape[::-1] != new_unpad:
        wmap = cv2.resize(wmap, new_unpad, interpolation=cv2.INTER_NEAREST)

    # Center padding
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    wmap = cv2.copyMakeBorder(wmap, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=pad_value)

    return wmap

class WaferDataset_utils:
    @staticmethod
    def transform_three_channels(img: np.ndarray) -> np.ndarray:
        ch0 = (img == 0).astype(np.float32)
        ch1 = (img == 1).astype(np.float32)
        ch2 = (img == 2).astype(np.float32)
        return np.stack([ch0, ch1, ch2], axis=0)
    
# Custom dataset
class WaferDataset(Dataset):
    def __init__(self, wmaps, labels):
        self.wmaps = np.array(wmaps) # [N, H, W]
        self.labels = np.array(labels) # [N]
    
    def __len__(self):
        return len(self.wmaps) 

    def __getitem__(self, idx):
        img = self.wmaps[idx] # [H, W]

        # One hot channel
        x = WaferDataset_utils.transform_three_channels(img) # [H, W] -> [3, H, W]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(self.labels[idx])
        return x, y


class AugmentedWaferDataset(Dataset):
    def __init__(self, wmaps, labels, target_counts : Optional[Dict] = None, augment_transform : Optional[Callable] = None, seed: int = 42):

        self.augment_transform = augment_transform
        self.target_counts = target_counts if target_counts is not None else {}
        self.class_data_dict = self._group_by_class(wmaps, labels)        
        self.augmented_data, self.augmented_labels = self._build_balanced_dataset()

    def __len__(self):
        return len(self.augmented_data)

    def __getitem__(self, idx):
        img, y = self.augmented_data[idx], self.augmented_labels[idx]
        x = WaferDataset_utils.transform_three_channels(img) # [1, H, W] -> # [3, H, W]
        x = torch.tensor(x, dtype=torch.float32)        
        y = torch.tensor(y)
        return x, y

    def _group_by_class(self, wmaps, labels):
        class_data_dict = {label: [] for label in set(labels)}

        for wmap, label in zip(wmaps, labels):
            class_data_dict[label].append(wmap)
        return class_data_dict # dict{0: [data...], 1: [data...], ... ,8: [data...]}


    def _build_balanced_dataset(self):
        """Generate a balanced dataset across all classes"""

        augmented_data, augmented_labels = [], []

        for label, wmap_list in self.class_data_dict.items():
            d, l = self._balance_class(label, wmap_list)
            augmented_data.extend(d)
            augmented_labels.extend(l)
        return augmented_data, augmented_labels
    
    def _balance_class(self, label, wmap_list):
        """Augment a specific class up to the target count"""    

        augmented_data, augmented_labels = [], []
        cur_count = len(wmap_list)
        target_count = self.target_counts.get(label, cur_count)

        # add original data
        augmented_data.extend(wmap_list)
        augmented_labels.extend([label] * cur_count)

        # add original data
        need_counts = max(0, target_count - cur_count)
        for _ in range(need_counts):
            wmap_sample = random.choice(wmap_list)
            augmented_data.append(self._augment_sample(wmap_sample))
            augmented_labels.append(label)        
        return augmented_data, augmented_labels 
    
    def _augment_sample(self, wmap_sample):
        x = torch.tensor(wmap_sample, dtype=torch.float32) # [H, W]
        
        if self.augment_transform:
            x = x.unsqueeze(0) # [H, W] -> [C=1, H, W]
            x = self.augment_transform(x) # Augmentation
            x = x.squeeze(0) # [C=1, H, W] -> [H, W]
        return x.squeeze(0).numpy()