from typing import List, Dict, Any, Iterator

import wandb
from torch import Tensor

import pytorch_lightning as pl
from pytorch_lightning import Callback, Trainer, LightningModule

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.figure import Figure

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

class Reconstructor(Callback):
    def __init__(self,
                 reconstruct_step: int = 40,
                 perc_recon: float = 0.25
                 ) -> None:
        super().__init__()
        '''
        A callback to reconstruct the reconstruction of the original
        reconstruct_step: int -> training steps before reconstructions are produced
        perc_recon: float -> percentage of the batch to be reconstructed
        '''

        self.reconstruct_step = reconstruct_step
        self.perc_recon = perc_recon


    # def on_train_batch_end(self,
    #                        trainer: Trainer,
    #                        pl_module: LightningModule,
    #                        outputs: Dict[str, Tensor],
    #                        batch: Dict[str, Tensor],
    #                        batch_idx: int) -> None:

    #     # recreate every 40 global steps
    #     if trainer.global_step % self.reconstruct_step == 0:
    #         # this import here stopped the psutil error, and tkinker error of image not being in main thread
    #         from matplotlib import pyplot as plt
    #         for fig in self.generate_figures(trainer, outputs):
    #             pl_module.logger.experiment.log({"train/image": wandb.Image(fig) })
    #             plt.close(fig)

    def on_validation_epoch_start(self,
                             trainer: pl.Trainer,
                             pl_module: pl.LightningModule,
                             ) -> None:
        self.val_steps = 0

    def on_validation_batch_end(self,
                                trainer: Trainer,
                                pl_module: LightningModule,
                                outputs: Dict[str, Tensor],
                                batch: Dict[str, Tensor],
                                batch_idx: int,
                                ) -> None:

        if self.val_steps % self.reconstruct_step == 0:
            from matplotlib import pyplot as plt
            for fig in self.generate_figures(trainer ,outputs):
                pl_module.logger.experiment.log({"val/image": wandb.Image(fig) })
                plt.close(fig)

        self.val_steps += 1

    def on_test_epoch_start(self,
                             trainer: pl.Trainer,
                             pl_module: pl.LightningModule,
                             ) -> None:
        self.test_steps = 0

    def on_test_batch_end(self,
                          trainer: Trainer,
                          pl_module: LightningModule,
                          outputs: Dict[str, Tensor],
                          batch: Dict[str, Tensor],
                          batch_idx: int,
                          ) -> None:

        if self.test_steps % self.reconstruct_step == 0:
            from matplotlib import pyplot as plt
            for fig in self.generate_figures(trainer ,outputs):
                pl_module.logger.experiment.log({"test/image": wandb.Image(fig) })
                plt.close(fig)

        self.test_steps += 1


    def generate_figures(self,
                   trainer: Trainer,
                   outputs: Dict[str, Tensor]) -> Iterator[Figure]:

        # loading  modules here stops internel rm tree os error and tkinter main thread error
        from matplotlib import pyplot as plt
        from matplotlib import gridspec as gs
        import numpy as np

        # extract outputs required
        xs, x_hats = [outputs[key] for key in ["x", "x_hat"]]


        # print(xs[0,3:6,:,:])
        num_rows = 2
        mask = np.random.randint(0, xs.size(0), int(xs.size(0)*self.perc_recon)) # take a percentage of the batch

        # create and yield a figure for each observation
        for j, (x, x_hat) in enumerate(zip(xs[mask], x_hats[mask])):

            # reverse the normalisation
            x = x.cpu() #* trainer.datamodule.std.view(-1, 1, 1) + trainer.datamodule.mean.view(-1, 1, 1)
            x_hat = x_hat.cpu() #* trainer.datamodule.std.view(-1, 1, 1) + trainer.datamodule.mean.view(-1, 1, 1)
            # create a figure and grid spec
            fig = plt.figure(figsize=(11.69, 8.27), dpi=100)
            grid_spec = gs.GridSpec(num_rows, 2,
                                    width_ratios=[1, 0.04],
                                    hspace=0.3)

            # plot observation, reconstruction and uncertainty image
            ax1 = fig.add_subplot(grid_spec[0, 0])
            ax2 = fig.add_subplot(grid_spec[1, 0])

            # plot input
            ax1.set_title("Input HSI (RGB only)", fontsize='medium')
            x = x[3:6,:,:].cpu().numpy()
            ax1.imshow(x)

            # plot reconstruction
            ax2.set_title("Reconstructed HSI (RGB only)", fontsize='medium')
            x_hat = x_hat[3:6,:,:].cpu().numpy()
            ax2.imshow(x_hat)

            # set a title
            # suptitle = ", ".join(self.get_label_names(y.cpu(), trainer.datamodule.label_decoder))
            # fig.suptitle(suptitle, wrap=True)

            # yield fig
            plt.show()


    def correct_img(self, img):
        img = self.normalise(img) # to clip between bounds
        img = np.moveaxis(img, 0, 2) # C, W, H > W, H, C
        img = img[:,:,::-1] # BGR to RGB
        return img

    def normalise(self, img):
        for band in range(len(img)):
            # normalize based on min/max pixel values to clamp ranges in [0, 1]
            img[band, ...] = (img[band, ...] - np.min(img[band, ...])) / np.max(img[band, ...])
        return img

    def get_label_names(self, onehots, class_names):
        """get all the names when a label is a one-hot."""
        label_idx = np.where(onehots)[0]
        label_names = [class_names[19][idx] for idx in label_idx]
        return label_names
