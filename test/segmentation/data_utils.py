import torch as th
import os
import json as js
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random as rd

from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.utils import (
    draw_bounding_boxes,
    draw_segmentation_masks
)
from torchvision.transforms.v2 import (
    Normalize,
    Resize,
    Compose
)
from torchvision.transforms.functional import to_pil_image


class SegmentationCOCOSet(Dataset):

    def __init__(
        self,
        data_dir: str,
        data_split: str = "train",
        images_size: int | tuple[int] = (128, 128),
        norm_mean: list[float] = (0.12, 0.23, 0.34),
        norm_std: list[float] = (0.12, 0.23, 0.34)
    ) -> None:
        
        
        self.tf = Compose([
            # Normalize(mean=norm_mean, std=norm_std),
            Resize(images_size)
        ])

        self.data_dir = os.path.join(data_dir, data_split)
        with open(os.path.join(self.data_dir, "_annotations.coco.json"), "r") as json_file:

            self.data = js.load(json_file)
            self.data["images"] = {
                sample["id"]: sample 
                for sample in self.data["images"]
            }
        
        self.images_size = images_size
        (self.x_scale, self.y_scale) = self._get_scale_factor()

    
    def _get_scale_factor(self) -> tuple[float]:

        if not isinstance(self.images_size, tuple):

            max_x = float(max(self.data["images"][0]["width"], self.images_size))
            min_x = float(min(self.data["images"][0]["width"], self.images_size))
            max_y = float(max(self.data["images"][0]["height"], self.images_size))
            min_y = float(min(self.data["images"][0]["height"], self.images_size))
        

        else:

            max_x = float(max(self.data["images"][0]["width"], self.images_size[0]))
            min_x = float(min(self.data["images"][0]["width"], self.images_size[0]))
            max_y = float(max(self.data["images"][0]["height"], self.images_size[1]))
            min_y = float(min(self.data["images"][0]["height"], self.images_size[1]))
        
        return  (
            (min_x / max_x), 
            (min_y / max_y)
        )

    def _pts_to_mask(self, pts: np.ndarray) -> th.Tensor:
        
        pts = np.concatenate([
            pts[0:pts.shape[0]:2] * self.x_scale,
            pts[1:pts.shape[0]:2] * self.y_scale
        ], axis=1).astype(np.int32)

        if isinstance(self.images_size, tuple):
            mask = np.zeros(self.images_size)
        else:
            mask = np.zeros((self.images_size, self.images_size))

        return th.Tensor(cv2.fillPoly(mask.astype(np.uint8), pts=[pts], color=1))

    def _build_segmentation_mask(self, points: np.ndarray | dict[np.ndarray]) -> th.Tensor.bool:

        if isinstance(points, dict):
            
            masks = th.zeros((self.classes_n, ) + self.images_size)
            for cat_id in points.keys(): 
                masks[cat_id] = self._pts_to_mask(points[cat_id])
                
            return masks

        else:
            return self._pts_to_mask(points)

        

    
    @property
    def classes_n(self) -> int:
        return len(self.data["categories"])

    @property
    def _images_size(self) -> tuple[int]:
        
        if isinstance(self.images_size, tuple): return self.images_size
        else: return (self.images_size, self.images_size)

    def __len__(self) -> int:
        return len(self.data["images"])

    def __getitem__(self, idx) -> tuple[th.Tensor]:

        image = self.tf(read_image(os.path.join(
            self.data_dir, 
            self.data["images"][idx]["file_name"]
        )) / 256.0)
        segmentation_points = {
            sample["category_id"]: np.expand_dims(np.asarray(sample["segmentation"][0]), axis=-1) 
            for sample in self.data["annotations"] 
            if sample["image_id"] == idx
        }
        segmentation_masks = self._build_segmentation_mask(segmentation_points)

        return (
            image,
            segmentation_masks
        )





    
        
        
        
        
        