import torch as th
import os
import numpy as np
import matplotlib.pyplot as plt
import tqdm
plt.style.use("dark_background")

from torch.optim import Adam
from torch.nn import MSELoss

from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from seg_pc_dataset import SSPCDataset
from cnn_unet_model import UnetSegmenter


class PCDModelTrainerL:

    def __init__(
        self,
        model: Module,
        loss: Module,
        optim: Optimizer,
        loader: DataLoader,
        target_folder: str = None,
        batch_size: int = 32,
        device: str = "cpu",
        save_out_samples_png: bool = True,
        save_out_samples_pcd: bool = True,
    ) -> None:
        
        super().__init__()
        self.target_folder = target_folder
        self.model = model
        self.loss = loss
        self.optim = optim
        self.loader = loader
        self.device = device

        self.save_png = save_out_samples_png
        self.save_pcd = save_out_samples_pcd
        self.batch_size = batch_size

        if self.target_folder is not None:
            if not os.path.exists(self.target_folder):
                os.mkdir(self.target_folder)
                self._generation_folder_ = os.path.join(
                    self.target_folder,
                    "training_samples"
                )
                os.mkdir(self._generation_folder_)


    def _save_generation_(self, epoch: int) -> None:

        if self.target_folder is not None:
            save_path = os.path.join(
                self._generation_folder_,
                f"Samples_at{epoch}epoch"
            )
        x, _ = next(iter(self.loader))
        x = x[th.randint(0, 32, (1, ))].to(self.device)
        

        if (self.save_png) and (self.target_folder is not None):
            
            png_out = self.model(x).permute((0, 2, 3, 1)).squeeze(dim=0)
            png_path = os.path.join(
                save_path,
                "samples.png"
            )
            fig, axis = plt.subplots()
            axis.imshow(png_out)
            fig.savefig(png_path)
        
        if (self.save_pcd) and (self.target_folder is not None):

            pcd_out, colors = self.model.predict(x)
            pcd_path = os.path.join(
                save_path,
                "pcd_samples.pt"
            )
            th.save((pcd_out, colors), pcd_path)
    
    
    def _train_on_epoch_(self, epoch: int) -> float:

        local_loss = 0.0
        for (pcd, pcd_sup) in tqdm.tqdm(
            self.loader, 
            colour="green",
            ascii=":>",
            desc=f"Epoch: {epoch}"
        ):
            
          
            self.optim.zero_grad()
            pcd = pcd.to(self.device)
            pcd_sup = pcd_sup.to(self.device)
            

            model_out = self.model(pcd)
            loss = self.loss(model_out, pcd_sup)
            local_loss += loss.item()
            
            loss.backward()
            self.optim.step()
        
        return local_loss
    
    def train(
        self,
        epochs: int = 10,
        epochs_per_save: int = 5
    ) -> dict[np.ndarray]:

        loss_buffer_ = []
        for epoch in range(epochs):
            
            local_loss = self._train_on_epoch_(epoch=epoch)
            loss_buffer_.append(local_loss)
            if (epoch % epochs_per_save) == 0:
                self._save_generation_(epoch=epoch)
        

        params_path = os.path.join(
            self._generation_folder_,
            "pcd_segmenter_params.pt"
        )
        th.save(self.model.state_dict(), params_path)
        return {
            "loss": np.asarray(loss_buffer_)
        }
            
            


        
        
