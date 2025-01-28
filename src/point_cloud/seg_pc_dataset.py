import numpy as np
import torch as th
import os
import open3d as o3d
import matplotlib.pyplot as plt
plt.style.use("dark_background")

from open3d.geometry import PointCloud
from open3d.utility import Vector3dVector
from open3d.visualization import Visualizer
from torch.utils.data import (
    Dataset,
    DataLoader
)



def _colorize_pcd_(
    path: str,
    labels_n: int
) -> None:

    labels_colors = {
        idx: np.random.random(3)
        for idx in range(labels_n)
    }
    velodyne_path = path
    labels_path = velodyne_path.replace("velodyne", "labels").replace(".bin", ".label")
    with open(velodyne_path, "rb") as vel_f:
        pcd = np.fromfile(vel_f, dtype=np.float32).reshape(-1, 5)[:, :3]
        
    with open(labels_path, "rb") as labels_f:
        labels = np.fromfile(labels_f, dtype=np.int32).reshape(-1)
    
    colors = np.asarray([labels_colors[label] for label in labels])
    return (pcd, colors)




        
class SSPCDataset(Dataset):


    def __init__(
        self,
        source_folder: str,
        split: str,
        scale_factor: float = None,
        angle: float = None,
        max_points: int = 300000,
        labels_n: int = 21,
        down_vert: int = 0.67,
        up_vert: int = 0.67,
        projection_width: int = 1024,
        projection_height: int = 256
    ) -> None:
        
        super().__init__()
        self.down_vert = down_vert
        self.up_vert = up_vert
        self.width = projection_width
        self.height = projection_height

        self._label_colors_ = {
            idx: np.random.random(3)
            for idx in range(labels_n)
        }
        self.angle = angle
        self.scale_factor = scale_factor
        self.max_points = max_points
        velodyne_root = os.path.join(
            source_folder,
            split,
            "velodyne"
        )
        self.velodyne_paths = [os.path.join(velodyne_root, path) for path in os.listdir(velodyne_root)]
        
        
    def _pc_to_plane_(
        self, 
        inputs: np.ndarray,
        colors: np.ndarray = None,
        with_cores: bool = False
    ) -> np.ndarray:

        pj_points = []
        for point in inputs:
            
            norm = np.linalg.norm(point[:3])
            vert = self.up_vert + self.down_vert
            u = (1 / 2) * (1 - np.arctan2(point[1], point[0]) * (1 / np.pi)) * self.width
            v = (1 - (np.arcsin(point[2] * (1 / norm)) + self.up_vert) * (1 / vert)) * self.height
            pj_points.append((v, u))

        pj_points = np.asarray(pj_points, dtype=np.int16)
        block_projection_ = np.zeros((self.height, self.width, 3))
        if with_cores and (colors is not None):
            block_projection_ = np.zeros((self.height, self.width, 6))
        

        for idx, angles in enumerate(pj_points):

            try:
                
                sample_ = inputs[idx]
                if with_cores and (colors is not None):
                    sample_ = np.concatenate([sample_, colors[idx]])
                
                elif (not with_cores) and (colors is not None):
                    sample_ = colors[idx]

                block_projection_[
                    angles[0], 
                    angles[1],
                    :
                ] = sample_
            
            except:
                pass

        return (block_projection_, pj_points)

    def _plane_to_pc_(inputs: np.ndarray) -> np.ndarray:

        pcd = []
        for row in inputs:
            for px in row:
                if np.linalg.norm(px) != 0:
                    pcd.append(px[:3])
        
        pcd = np.asarray(pcd)
        return pcd

    def __len__(self) -> int:
        return len(self.velodyne_paths)
    
    def __getitem__(self, idx) -> tuple[th.Tensor]:

        vel_path = self.velodyne_paths[idx]
        labels_path = vel_path.replace("velodyne", "labels").replace(".bin", ".label")
        
        with open(vel_path, "rb") as velodyne_f:

            pcd_ = np.fromfile(velodyne_f, dtype=np.float32).reshape(-1, 5)[:self.max_points, :4]
            if self.angle is not None:
                rot = np.array([
                    [np.cos(self.angle), np.sin(self.angle), 0],
                    [-np.sin(self.angle), np.cos(self.angle), 0],
                    [0, 0, 1]
                ])
                pcd_ = np.dot(pcd_, rot)
            
            if self.scale_factor is not None:
                pcd_ *= self.scale_factor
            
            pcd_, angles = self._pc_to_plane_(inputs=pcd_)
        
        with open(labels_path, "rb") as labels_f:

            labels_ = np.fromfile(labels_f, dtype=np.int32).reshape(-1)[:self.max_points]
            mask_ = np.zeros((self.height, self.width, 3))
            
            for label in np.unique(labels_):
                
                color = self._label_colors_[label]
                idx = angles[labels_[labels_ == label]]
                for px in idx:
                    print(px)
                    mask_[
                        px[0],
                        px[1],
                        :
                    ] = color
            
        
        pcd_ = th.Tensor(pcd_)
        mask_ = th.Tensor(mask_)
        return (
            pcd_,
            mask_
        )
        
        
if __name__ == "__main__":

    dataset = SSPCDataset(
        source_folder="C:\\Users\\1\\Desktop\\SemanticSTF",
        split="train"
    )   
    pcd, colors = _colorize_pcd_(
        path="C:\\Users\\1\\Desktop\\SemanticSTF\\train\\velodyne\\2018-02-04_11-09-42_00400.bin",
        labels_n=21
    )
    pcd = pcd[:dataset.max_points]
    colors = colors[:dataset.max_points]
    
    projection, _ = dataset._pc_to_plane_(
        inputs=pcd,
        colors=colors,
        with_cores=False
    )
    
    
    # loader = DataLoader(
    #     dataset=dataset,
    #     batch_size=32,
    #     shuffle=True
    # )
    # plt.style.use("dark_background")
    # pcd, mask = next(iter(loader))
    _, axis = plt.subplots()
    axis.imshow(projection)
    plt.show()
    


