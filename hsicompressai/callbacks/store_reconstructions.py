import os
import numpy as np
import torch
import pytorch_lightning as pl


class SaveReconstructedImagesCallback(pl.Callback):
    def __init__(self, save_dir, folder_name="reconstructions", max_images=5):
        super().__init__()
        self.folder_name = folder_name
        self.max_images = max_images
        self.save_dir = save_dir  # will be set later

    def setup(self, trainer, pl_module, stage=None):
        # Use the log directory from the logger (Hydra's working dir)
        # log_dir = trainer.logger.log_dir if hasattr(trainer.logger, "log_dir") else trainer.default_root_dir
        self.save_dir = os.path.join(self.save_dir, self.folder_name)
        os.makedirs(self.save_dir, exist_ok=True)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):

        inputs, _ = batch if isinstance(batch, (list, tuple)) else (batch, None)
        with torch.no_grad():
            recon = pl_module(inputs)

        # Move to CPU and convert to numpy
        inputs_np = inputs.detach().cpu().numpy()
        recon_np = recon.detach().cpu().numpy()

        # Save a few examples from the batch
        for i in range(inputs_np.shape[0]):
            np.save(os.path.join(self.save_dir, f"input_{batch_idx}_{i}.npy"), inputs_np[i])
            np.save(os.path.join(self.save_dir, f"recon_{batch_idx}_{i}.npy"), recon_np[i])
