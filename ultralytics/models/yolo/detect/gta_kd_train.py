# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Train a model on a dataset.

Usage:
    $ yolo mode=train model=yolo26n.pt data=coco8.yaml imgsz=640 epochs=100 batch=16
"""

from __future__ import annotations

import math
import time
import warnings

import numpy as np
import torch
from torch import distributed as dist

from ultralytics.utils import (
    DEFAULT_CFG,
    LOGGER,
    RANK,
    TQDM,
    colorstr,
)
from ultralytics.utils.torch_utils import (
    autocast,
    unset_deterministic,
    unwrap_model,
)
from ultralytics.utils.gta_kd_loss import GTADistillationLoss
from .kd_train import KD_Trainer

class GTA_KD_Trainer(KD_Trainer):
    """A class extending the KD_Trainer class for training Ground Truth Aware + Class Mapping Knowledge Distillation (KD)."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        super().__init__(cfg, overrides, _callbacks)
        self.class_mapping = overrides.get("class_mapping", None)
        self.teacher_pred_conf = overrides.get("teacher_pred_conf", 0.01)
        
    def _do_train(self):
        """Perform the full training loop including setup, epoch iteration, validation, and final evaluation."""
        if self.world_size > 1:
            self._setup_ddp()
        self._setup_train()

        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
            f"Using {self.train_loader.num_workers * (self.world_size or 1)} dataloader workers\n"
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f"Starting training for " + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
        )
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
                # make loss
        if self.teacher is not None:
            distillation_loss = GTADistillationLoss(self.model, self.teacher, distiller=self.loss_type, distill_loss_weight=self.distill_loss_weight, s_layers=self.s_layers, t_layers=self.t_layers, class_mapping=self.class_mapping, teacher_pred_conf=self.teacher_pred_conf)
        
        epoch = self.start_epoch
        self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start
        self._oom_retries = 0  # OOM auto-reduce counter for first epoch
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()

            self._model_train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
                        
            if self.teacher is not None:
                distillation_loss.register_hook()

            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for x in self.optimizer.param_groups:
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(
                            ni,
                            xi,
                            [
                                self.args.warmup_bias_lr if x.get("param_group") == "bias" else 0.0,
                                x["initial_lr"] * self.lf(epoch),
                            ],
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # Forward
                try:
                    with autocast(self.amp):
                        batch = self.preprocess_batch(batch)
                        if self.args.compile:
                            # Decouple inference and loss calculations for improved compile performance
                            preds = self.model(batch["img"])
                            loss, self.loss_items = unwrap_model(self.model).loss(batch, preds)
                        else:
                            loss, self.loss_items = self.model(batch)
                        self.loss = loss.sum()
                        if RANK != -1:
                            self.loss *= self.world_size

                    # Add more distillation logic
                    if self.teacher is not None:
                        if self.cos_d_loss:
                            distill_weight = ((1 - math.cos(i * math.pi / len(self.train_loader))) / 2) * (0.1 - 1) + 1
                        else: 
                            ratio = 1 - (i / len(self.train_loader))
                            distill_weight = (0.1 + 0.9 * ratio)
                        with torch.no_grad():
                            pred = self.teacher(batch['img'])
                            
                        self.d_loss = distillation_loss.get_loss(batch, pred)
                        self.d_loss *= distill_weight
                        self.loss += self.d_loss
                        self.loss_items = torch.cat((self.loss_items, self.d_loss.detach().view(1)))
                    
                    self.tloss = (
                        self.loss_items if self.tloss is None else (self.tloss * i + self.loss_items) / (i + 1)
                    )
                    # Backward
                    self.scaler.scale(self.loss).backward()
                except torch.cuda.OutOfMemoryError:
                    if epoch > self.start_epoch or self._oom_retries >= 3 or RANK != -1:
                        raise  # only auto-reduce during first epoch on single GPU, max 3 retries
                    self._oom_retries += 1
                    old_batch = self.batch_size
                    self.args.batch = self.batch_size = max(self.batch_size // 2, 1)
                    LOGGER.warning(
                        f"CUDA out of memory with batch={old_batch}. "
                        f"Reducing to batch={self.batch_size} and retrying ({self._oom_retries}/3)."
                    )
                    self._clear_memory()
                    self._build_train_pipeline()  # rebuild dataloaders, optimizer, scheduler
                    self.scheduler.last_epoch = self.start_epoch - 1
                    nb = len(self.train_loader)
                    nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1
                    last_opt_step = -1
                    self.optimizer.zero_grad()
                    break  # restart epoch loop with reduced batch size
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    # Timed stopping
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:  # if DDP training
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                            self.stop = broadcast_list[0]
                        if self.stop:  # training time exceeded
                            break

                # Log
                if RANK in {-1, 0}:
                    loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            f"{self._get_memory():.3g}G",  # (GB) GPU memory util
                            *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),  # losses
                            batch["cls"].shape[0],  # batch size, i.e. 8
                            batch["img"].shape[-1],  # imgsz, i.e 640
                        )
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")
                if self.stop:
                    break  # allow external stop (e.g. platform cancellation) between batches
            else:
                # for/else: this block runs only when the for loop completes without break (no OOM retry)
                self._oom_retries = 0  # reset OOM counter after successful first epoch
            
            if self._oom_retries and not self.stop:
                continue  # OOM recovery broke the for loop, restart with reduced batch size

            if hasattr(unwrap_model(self.model).criterion, "update"):
                unwrap_model(self.model).criterion.update()
                
            # More distillation logic
            if self.teacher is not None:
                distillation_loss.remove_handle_()

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers

            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

            # Validation
            final_epoch = epoch + 1 >= self.epochs
            if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                self._clear_memory(None if self.device.type == "mps" else 0.5)  # prevent VRAM spike
                self.metrics, self.fitness = self.validate()

            # NaN recovery
            if self._handle_nan_recovery(epoch):
                continue

            self.nan_recovery_attempts = 0
            if RANK in {-1, 0}:
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                # Save model
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

            # Scheduler
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch  # do not move
                self.stop |= epoch >= self.epochs  # stop if exceeded epochs
            self.run_callbacks("on_fit_epoch_end")
            # clear if memory utilization > 50%; always clear on MPS due to leak https://github.com/ultralytics/ultralytics/issues/22621
            self._clear_memory(None if self.device.type == "mps" else 0.5)

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks
            epoch += 1

        seconds = time.time() - self.train_time_start
        LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
        # Do final val with best.pt
        self.final_eval()
        if RANK in {-1, 0}:
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        unset_deterministic()
                
        # Distill logic
        if self.teacher is not None:
            distillation_loss.remove_handle_()

        self.run_callbacks("teardown")