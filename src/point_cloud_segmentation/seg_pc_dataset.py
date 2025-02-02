import numpy as np
import torch as th
import os
import open3d as o3d
import matplotlib.pyplot as plt
import tqdm 
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
    
    pcd_o3d = PointCloud()
    pcd_o3d.points = Vector3dVector(pcd)
    pcd_o3d.colors = Vector3dVector(colors)
    vis = Visualizer()
    vis.create_window(visible=True)
    vis.get_render_option().background_color = [0, 0, 0]
    vis.add_geometry(pcd_o3d)
    vis.run()
    

class SSPCDatFormater():


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
        projection_width: int = 1023,
        projection_height: int = 255,
        masking_with_coordinates: bool = True
    ) -> None:
        
        super().__init__()
        self.with_cores_ = masking_with_coordinates
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
        self.target_root = os.path.join(
            source_folder,
            split
        )
        for split_tar in ["velodyne_unlabeled", "velodyne_labeled"]:
            split_root = os.path.join(
                self.target_root,
                split_tar
            )
            if not os.path.exists(split_root):
                os.mkdir(split_root)

        velodyne_root = os.path.join(
            source_folder,
            split,
            "velodyne"
        )
        self.velodyne_paths = [os.path.join(velodyne_root, path) for path in os.listdir(velodyne_root)]
        self._formate_data_()
    
    def _pc_to_plane_(
        self, 
        inputs: np.ndarray,
        colors: np.ndarray = None,
    ) -> np.ndarray:

        pj_points = []
        for point in inputs:
            
            norm = np.linalg.norm(point)
            vert = self.up_vert + self.down_vert
            u = (1 - (np.arcsin(point[2] * (1 / norm)) + self.up_vert) * (1 / vert)) * self.height
            v = (1 / 2) * (1 - np.arctan2(point[1], point[0]) * (1 / np.pi)) * self.width
            pj_points.append((u, v))

        pj_points = np.asarray(pj_points, dtype=np.int16)
        block_projection_ = np.zeros((self.height, self.width, 3))
        if self.with_cores_ and (colors is not None):
            block_projection_ = np.zeros((self.height, self.width, 6))
        

        for idx, angles in enumerate(pj_points):
            
            sample_ = inputs[idx]
            if self.with_cores_ and (colors is not None):
                sample_ = np.concatenate([colors[idx], sample_])
            
            elif (not self.with_cores_) and (colors is not None):
                sample_ = colors[idx]
            
            try:
                block_projection_[
                    angles[0], 
                    angles[1],
                    :
                ] = sample_
            
            except:
                pass

        return block_projection_
    

    def _formate_data_(self) -> None:

        for idx, vel_path in enumerate(tqdm.tqdm(
            self.velodyne_paths,
            desc="Formating Data",
            colour="green",
            ascii=":>"
        )):

            with open(vel_path, "rb") as velodyne_f:

                pcd_ = np.fromfile(velodyne_f, dtype=np.float32).reshape(-1, 5)[:self.max_points, :3]
                if self.angle is not None:
                    rot = np.array([
                        [np.cos(self.angle), np.sin(self.angle), 0],
                        [-np.sin(self.angle), np.cos(self.angle), 0],
                        [0, 0, 1]
                    ])
                    pcd_ = np.dot(pcd_, rot)
                
                if self.scale_factor is not None:
                    pcd_ *= self.scale_factor
                
                pcd_pj_ = self._pc_to_plane_(inputs=pcd_)
            
            labels_path = vel_path.replace("velodyne", "labels").replace(".bin", ".label")
            with open(labels_path, "rb") as labels_f:

                labels_ = np.fromfile(labels_f, dtype=np.int32).reshape(-1)[:self.max_points]
                colors_ = np.asarray([
                    self._label_colors_[label]
                    for label in labels_
                ])[:self.max_points]
                mask_ = self._pc_to_plane_(pcd_, colors=colors_)
                

            pcd_pj_ = th.Tensor(pcd_pj_).permute((-1, 0, 1))
            mask_ = th.Tensor(mask_).permute((-1, 0, 1))
            
            tar_path_unlabeled = os.path.join(
                self.target_root,
                "velodyne_unlabeled",
                f"Sample{idx}.pt"
            )
            tar_path_labeled = tar_path_unlabeled.replace("unlabeled", "labeled")
            th.save(pcd_pj_, tar_path_unlabeled)
            th.save(mask_, tar_path_labeled)
        


class SSPCDataset(Dataset):

    def __init__(
        self,
        source_folder: str,
        split: str
    ):

        super().__init__()
        velodyne_unlabeled = os.path.join(
            source_folder,
            split,
            "velodyne_unlabeled"
        )
        self.paths_ = [
            os.path.join(velodyne_unlabeled, path)
            for path in os.listdir(velodyne_unlabeled)
        ]
    
    def __len__(self) -> None:
        return len(self.paths_)

    def __getitem__(self, idx: int) -> tuple[th.Tensor]:

        unlabeled_path = self.paths_[idx]
        labeled_path = unlabeled_path.replace("velodyne_unlabeled", "velodyne_labeled")
        
        return (
            th.load(unlabeled_path, weights_only=True),
            th.load(labeled_path, weights_only=True)
        )
        
        






