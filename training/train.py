from typing import Any, Dict, List, Optional, Tuple, Type

import hydra
from copy import deepcopy

import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger
from omegaconf import DictConfig

from hsicompressai.utils import utils
from hsicompressai.utils import get_pylogger

log = get_pylogger(__name__)

@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    #if cfg.get("seed"):
    #    pl.seed_everything(cfg.seed, workers=True)

    #log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    #datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    #datamodule.prepare_data()  # download if not present
    #datamodule.setup()

    #log.info(f"Instantiating model <{cfg.model._target_}>")
    #model: LightningModule = hydra.utils.instantiate(cfg.model)

    #log.info("Instantiating callbacks...")
    #callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    #log.info("Instantiating loggers...")
    #logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    #log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    #trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    #object_dict = {
    #    "cfg": cfg,
    #    "datamodule": datamodule,
    #    "model": model,
    #    "callbacks": callbacks,
    #    "logger": logger,
    #    "trainer": trainer,
    #}

    #if logger:
    #    log.info("Logging hyperparameters!")
    #    utils.log_hyperparameters(object_dict)

    #if cfg.get("train"):
    #    log.info("Starting training!")
    #    # datamodule.setup_folds(cfg.get("num_folds"))
    #    # store the original state of the 1st generated model for continuity across the folds
    #    # lightning_module_state_dict = deepcopy(model.state_dict())
    #    # current_fold=0

    #    # for current_fold in range(cfg.get("num_folds")):
    #    # log.info(f"Current fold {current_fold}")
    #    # datamodule.setup_fold_index(current_fold)
    #    # model.load_state_dict(lightning_module_state_dict)

    #    #TODO update the checkpoint exportpath in model checkpoint to include fold
    #    # export_path = cfg.get("callbacks")['model_checkpoint']['dirpath']

    #    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
    #    train_metrics = trainer.callback_metrics

    #        # trainer.save_checkpoint(osp.join(export_path, f"model.{self.current_fold}.pt"))

    #    # checkpoint_paths = [osp.join(self.export_path, f"model.{f_idx + 1}.pt") for f_idx in range(cfg.get("folds"))]


    #if cfg.get("test"):
    #    log.info("Starting testing!")
    #    ckpt_path = trainer.checkpoint_callback.best_model_path
    #    if ckpt_path == "":
    #        log.warning("Best ckpt not found! Using current weights for testing...")
    #        ckpt_path = None
    #    trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    #    log.info(f"Best ckpt path: {ckpt_path}")

    #test_metrics = trainer.callback_metrics

    #metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
