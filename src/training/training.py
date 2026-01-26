import argparse
import datetime
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    StepLR,
    CosineAnnealingLR,
    OneCycleLR,
)
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend

from data_sets.dsets import AutoQcDataset
from inference.make_predictions import (
    add_predicted_values,
    compute_standardized_rmse,
    create_correlation_coefficient,
    create_scatter_plot,
    get_validation_info,
)
from models.torchmodels import AlexNet3D
from models.regressor import get_regressor_model
from util.logconf import logging
from util.util import enumerateWithEstimate


# This script is a comprehensive deep learning pipeline for training a model to
# predict QU_motion scores from MRI scans. It includes data loading, model selection,
# training/validation loops, logging, and model saving.


log = logging.getLogger(__name__)

# Used for computeBatchLoss and logMetrics to index into metrics_t/metrics_a
METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_LOSS_NDX = 2
METRICS_SIZE = 3

def count_items(input_list):
    counts = {}
    for item in input_list:
        if item in counts:
            counts[item] += 1
        else:
            counts[item] = 1
    return counts

# Refactored Configuration class to handle CLI arguments
class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument(
            "--tb-prefix", default="qu_motion", help="Tensorboard data prefix."
        )
        self.parser.add_argument("--csv-input-file", help="CSV data file.")
        self.parser.add_argument(
            "--num-workers", default=8, type=int, help="Number of worker processes"
        )
        self.parser.add_argument(
            "--batch-size", default=32, type=int, help="Batch size for training"
        )
        self.parser.add_argument(
            "--epochs", default=1, type=int, help="Number of epochs to train"
        )
        # self.parser.add_argument(
        #     "--subset-power",
        #     default=None,
        #     type=int,
        #     help="Load only a subset of the dataset as a power of 2 (e.g., 3 for 2^3=8 samples). If None, use full dataset."
        # )
        self.parser.add_argument(
            "--model-save-location",
            default=f"./model-{datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')}.pt",
        )
        self.parser.add_argument("--plot-location", help="Location to save plot")
        self.parser.add_argument("--optimizer", default="Adam", help="Optimizer type.")
        self.parser.add_argument(
            "--model",
            default="Regressor",
            help="Model type",
            choices={"AlexNet", "Regressor"},
        )
        self.parser.add_argument(
            "comment", nargs="?", default="dcan", help="Comment for Tensorboard run"
        )
        self.parser.add_argument(
            "--lr", default=0.001, type=float, help="Learning rate"
        )
        #self.parser.add_argument("--gd", type=int, help="Use Gd-enhanced scans.")
        self.parser.add_argument("--use-train-validation-cols", action="store_true")
        self.parser.add_argument("--folder", help="Folder where MRIs are stored")
        self.parser.add_argument("--csv-output-file", help="CSV output file.")
        self.parser.add_argument("--use-weighted-loss", action="store_true")
        self.parser.add_argument(
            "--scheduler",
            default="plateau",
            choices=["plateau", "step", "cosine", "onecycle"],
            help="Learning rate scheduler",
        )
        self.parser.add_argument(
            "--train-split",
            default=0.8,
            type=float,
            help="Training split ratio (default: 0.8)",
        )
        self.parser.add_argument(
            "--split-strategy",
            default="random",
            choices=["random", "stratified", "sequential"],
            help="Strategy for train/validation split",
        )
        self.parser.add_argument(
            "--random-seed",
            default=42,
            type=int,
            help="Random seed for reproducible splits",
        )
        self.parser.add_argument(
            "--DEBUG", action="store_true", help="Set this flag to run in debug mode"
        )
        self.parser.add_argument(
            "--tb-run-dir", default="runs", help="Tensorboard log directory."
        )
        self.parser.add_argument(
            "--use-amp", action="store_true", help="Use Automatic Mixed Precision (AMP) for training."
        )
        self.parser.add_argument(
            "--use-kfold", action="store_true", help="Use k-fold cross-validation for training."
        )
        self.parser.add_argument(
            "--k-folds", default=5, type=int, help="Number of folds for k-fold cross-validation (default: 5)"
        )

    def parse_args(self, sys_argv: list[str]) -> argparse.Namespace:
        return self.parser.parse_args(sys_argv)
    



# Data Handler Class to manage dataset operations
class DataHandler:
    def __init__(self, df, output_df, use_cuda, batch_size, num_workers):
        self.df = df
        self.output_df = output_df
        self.use_cuda = use_cuda
        self.batch_size = batch_size
        self.num_workers = num_workers

    def init_dl(self, folder, subjects, is_val_set: bool = False):
        dataset = AutoQcDataset(
            folder, subjects, self.df, self.output_df, is_val_set_bool=is_val_set
        )
        batch_size = self.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.use_cuda,
        )

        return dataloader


# Model Handler Class to manage model operations
class ModelHandler:
    def __init__(self, model_name, use_cuda, device):
        self.model_name = model_name
        self.use_cuda = use_cuda
        self.device = device
        self.model = self._init_model()

    def _init_model(self):
        if self.model_name == "Regressor":
            model = get_regressor_model()
            log.info("Using Regressor model")
        else:
            model = AlexNet3D(4608)
            log.info("Using AlexNet3D")

        # Always move to the specified device consistently
        log.info(f"Moving model to device: {self.device}")
        model = model.to(self.device)

        # Apply DataParallel only if using CUDA and multiple GPUs
        if self.use_cuda and torch.cuda.device_count() > 1:
            log.info("Using CUDA with {} devices.".format(torch.cuda.device_count()))
            model = nn.DataParallel(model)

        return model

    def save_model(self, save_location):
        if isinstance(self.model, torch.nn.DataParallel):
            self.model = self.model.module
        try:
            torch.save(self.model.state_dict(), save_location)
            log.info(f"Model saved successfully to {save_location}")
        except Exception as e:
            log.error(f"Failed to save model to {save_location}: {e}")
            raise

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))



# Training/Validation Loop Handler
class TrainingLoop:
    def __init__(self, model_handler, optimizer, device, df, config, scheduler=None):
        self.model_handler = model_handler
        self.optimizer = optimizer
        self.device = device
        self.total_samples = 0
        self.df = df
        self.scheduler = scheduler
        self.config = config
        
        device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        self.scaler = GradScaler() if self.config.use_amp and device_type == 'cuda' else None
        self.device_type = device_type

        training_df = df[df["training"] == 1]
        qu_motion_scores = list(training_df["QU_motion"])
        item_counts = count_items(qu_motion_scores)
        weighted_counts = {key: 1.0 / value for key, value in item_counts.items()}
        self.weights = weighted_counts

    def train_epoch(self, epoch, train_dl):
        self.model_handler.model.train()
        trn_metrics_g = torch.zeros(
            METRICS_SIZE, len(train_dl.dataset), device=self.device
        )
        for batch_ndx, batch_tup in enumerateWithEstimate(
            train_dl, f"E{epoch} Training", start_ndx=train_dl.num_workers
        ):
            self.optimizer.zero_grad()
            
            if self.config.use_amp and self.scaler is not None:
                # Compute loss with autocast for mixed precision
                input_t, label_t, _, _ = batch_tup
                input_g = input_t.to(self.device, non_blocking=True)
                label_g = label_t.to(self.device, non_blocking=True)
                
                with autocast(self.device_type):
                    outputs_g = self.model_handler.model(input_g)
                    if isinstance(outputs_g, (list, tuple)):
                        outputs_g = outputs_g[0]
                    outputs_g = outputs_g.squeeze(dim=-1)
                    label_g_view = label_g.view(-1)
                    
                    if self.config.use_weighted_loss:
                        loss_g = self.weighted_mse_loss(outputs_g, label_g_view)
                        loss_var = loss_g.mean()
                    else:
                        loss_func = nn.MSELoss(reduction="none")
                        loss_g = loss_func(outputs_g, label_g_view)
                        loss_var = loss_g.mean()
                
                # Store metrics
                start_ndx = batch_ndx * train_dl.batch_size
                end_ndx = start_ndx + label_t.size(0)
                trn_metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = label_g_view.detach()
                trn_metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = outputs_g.detach()
                trn_metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss_var.detach()
                
                # Backward and optimizer step with scaled gradients
                self.scaler.scale(loss_var).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model_handler.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_var = self._compute_batch_loss(
                    batch_ndx, batch_tup, train_dl.batch_size, trn_metrics_g
                )
                loss_var.backward()
                self.optimizer.step()

            # Step OneCycleLR scheduler after each batch
            if self.scheduler and isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()

        self.total_samples += len(train_dl.dataset)
        return trn_metrics_g.to("cpu")
    
    def _compute_batch_loss_amp(self, batch_ndx, batch_tup, batch_size, metrics_g):
        """Compute batch loss with automatic mixed precision"""
        input_t, label_t, _, _ = batch_tup
        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        # Forward pass with autocast
        with autocast(self.device_type):
            outputs_g = self.model_handler.model(input_g)

            # If outputs_g is a list or tuple, take the first element
            if isinstance(outputs_g, (list, tuple)):
                outputs_g = outputs_g[0]

            outputs_g = outputs_g.squeeze(dim=-1)
            label_g = label_g.view(-1)

            # Calculate loss
            if self.config.use_weighted_loss:
                loss_g = self.weighted_mse_loss(outputs_g, label_g)
                loss_mean = loss_g.mean()
            else:
                loss_func = nn.MSELoss(reduction="none")
                loss_g = loss_func(outputs_g, label_g)
                loss_mean = loss_g.mean()

        # Store metrics (outside autocast)
        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)

        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = label_g.detach()
        metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = outputs_g.detach()
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss_mean.detach()

        return loss_mean

    def validate_epoch(self, epoch, val_dl):
        with torch.no_grad():
            self.model_handler.model.eval()
            val_metrics_g = torch.zeros(
                METRICS_SIZE, len(val_dl.dataset), device=self.device
            )
            for batch_ndx, batch_tup in enumerateWithEstimate(
                val_dl, f"E{epoch} Validation", start_ndx=val_dl.num_workers
            ):
                self._compute_batch_loss(
                    batch_ndx, batch_tup, val_dl.batch_size, val_metrics_g
                )
        return val_metrics_g.to("cpu")

    def weighted_mse_loss(self, predictions, targets):
        """
        Calculate weighted MSE loss for regression where weights are determined by
        the frequency of each target value in the training set.

        Args:
            predictions (torch.Tensor): Model predictions, shape [batch_size]
            targets (torch.Tensor): Ground truth values, shape [batch_size]

        Returns:
            torch.Tensor: Weighted MSE loss for each sample in the batch
        """
        # Convert targets to CPU for dictionary lookup if they're on GPU
        targets_cpu = targets.detach().cpu().numpy()

        # Create a tensor to store weights for each sample in the batch
        weights = torch.ones_like(predictions)

        # Assign weights based on target values
        for i, target in enumerate(targets_cpu):
            # Handle potential floating point issues by rounding
            target_key = round(float(target), 1)  # Adjust rounding precision as needed

            # Get weight from dictionary, default to 1.0 if not found
            if target_key in self.weights:
                weights[i] = torch.tensor(
                    self.weights[target_key], device=predictions.device
                )
            else:
                # For unseen values, use the mean weight or a default
                mean_weight = sum(self.weights.values()) / len(self.weights)
                weights[i] = torch.tensor(mean_weight, device=predictions.device)

        # Calculate weighted squared error for each sample
        squared_errors = (predictions - targets) ** 2
        weighted_squared_errors = weights * squared_errors

        return weighted_squared_errors

    def _compute_batch_loss(self, batch_ndx, batch_tup, batch_size, metrics_g):
        input_t, label_t, _, _ = batch_tup
        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        if self.config.DEBUG:
            logging.getLogger().setLevel(logging.DEBUG)
            log.setLevel(logging.DEBUG)
            logging.getLogger("data_sets").setLevel(logging.DEBUG)
            logging.getLogger("data_sets.dsets").setLevel(logging.DEBUG)
            log.debug(f"Input shape: {input_g.shape}")
            log.debug(f"Label shape: {label_g.shape}")
            log.debug(f"Input device: {input_g.device}")
            log.debug(
                f"Model device: {next(self.model_handler.model.parameters()).device}"
            )
            log.debug(f"Expected device: {self.device}")
        else:
            logging.getLogger().setLevel(logging.INFO)
            log.setLevel(logging.INFO)
            logging.getLogger("data_sets").setLevel(logging.INFO)
            logging.getLogger("data_sets.dsets").setLevel(logging.INFO)

        outputs_g = self.model_handler.model(input_g)

        # If outputs_g is a list or tuple, take the first element
        if isinstance(outputs_g, (list, tuple)):
            outputs_g = outputs_g[0]

        outputs_g = outputs_g.squeeze(dim=-1)  # Remove extra dimension if needed

        label_g = label_g.view(-1)  # Ensures shape is [batch_size]

        log.debug(f"outputs_g shape: {outputs_g.shape}")  # Should be [batch_size]
        log.debug(f"label_g shape: {label_g.shape}")  # Should be [batch_size]

        # When using Regressor for the model, it is important that we use nn.MSELoss for regression.
        if self.config.use_weighted_loss:
            loss_g = self.weighted_mse_loss(outputs_g, label_g)
            loss_mean = loss_g.mean()  # Get mean for backpropagation
        else:
            loss_func = nn.MSELoss(reduction="none")
            loss_g = loss_func(outputs_g, label_g)
            loss_mean = loss_g.mean()

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)

        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = label_g.detach()
        metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = outputs_g.detach()
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss_mean.detach()

        return loss_mean


# TensorBoard Logger
class TensorBoardLogger:
    def __init__(self, tb_prefix, time_str, comment):
        self.log_dir = os.path.join(tb_prefix, time_str)
        self.trn_writer = SummaryWriter(log_dir=self.log_dir + f"-trn_cls-{comment}")
        self.val_writer = SummaryWriter(log_dir=self.log_dir + f"-val_cls-{comment}")

    def log_metrics(self, mode_str, epoch, metrics, sample_count):
        writer = getattr(self, f"{mode_str}_writer")

        # Calculate actual loss from labels and predictions
        labels = metrics[METRICS_LABEL_NDX]
        predictions = metrics[METRICS_PRED_NDX]
        actual_losses = (labels - predictions) ** 2  # MSE per sample
        actual_mean_loss = actual_losses.mean()

        writer.add_scalar("loss/all", actual_mean_loss, sample_count)

        # Add correlation tracking
        correlation = torch.corrcoef(torch.stack([labels, predictions]))[0, 1]
        if not torch.isnan(correlation):
            writer.add_scalar("metrics/correlation", correlation, sample_count)

    def close(self):
        self.trn_writer.close()
        self.val_writer.close()


def get_folder_name(file_path):
    """
    Extracts the folder name from a given file path.

    Args:
      file_path: The path to the file.

    Returns:
      The name of the folder containing the file, or None if an error occurs.
    """
    try:
        folder_path = os.path.dirname(file_path)
        folder_name = os.path.basename(folder_path)
        return folder_name
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# Main Application Class
class AutoQcTrainingApp:
    def __init__(self, sys_argv=None):
        self.config = Config().parse_args(sys_argv)
        self.use_cuda = torch.cuda.is_available() and not self.config.DEBUG
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not self.config.DEBUG else "cpu"
        )
        self.model_handler = ModelHandler(self.config.model, self.use_cuda, self.device)
        self.optimizer = self._init_optimizer()
        self.tb_run_dir = self.config.tb_run_dir

        self.input_df = pd.read_csv(self.config.csv_input_file)
        self.output_df = self.input_df.copy()

        self.time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")

        self.data_handler = DataHandler(
            self.input_df,
            self.output_df,
            self.use_cuda,
            self.config.batch_size,
            self.config.num_workers,
        )
        self.folder = self.config.folder

        experiment_name = self.create_experiment_name()
        self.tb_logger = TensorBoardLogger(
            self.tb_run_dir, self.time_str, experiment_name
        )

    def create_experiment_name(self):
        """Create meaningful experiment name from hyperparameters"""
        parts = [
            f"model-{self.config.model}",
            f"lr-{self.config.lr}",
            f"bs-{self.config.batch_size}",
            f"epochs-{self.config.epochs}",
            f"sched-{self.config.scheduler}",
        ]

        if hasattr(self.config, "split_strategy"):
            parts.append(f"split-{self.config.split_strategy}")
            parts.append(f"ratio-{self.config.train_split:.1f}")

        if self.config.use_weighted_loss:
            parts.append("weighted")

        parts.append(self.time_str)

        return "_".join(parts)

    def _init_optimizer(self):
        optimizer_type = self.config.optimizer.lower()
        optimizer_cls = Adam if optimizer_type == "adam" else SGD
        return optimizer_cls(self.model_handler.model.parameters(), lr=self.config.lr)

    def _init_scheduler(self, train_dl):
        # '--scheduler', default='plateau',
        #    choices=['plateau', 'step', 'cosine', 'onecycle']
        if self.config.scheduler == "step":
            scheduler = StepLR(self.optimizer, step_size=30, gamma=0.1)
        elif self.config.scheduler == "cosine":
            scheduler = CosineAnnealingLR(self.optimizer, T_max=100, eta_min=0)
        elif self.config.scheduler == "onecycle":
            scheduler = OneCycleLR(
                self.optimizer,
                max_lr=0.01,
                total_steps=len(train_dl) * self.config.epochs,
                pct_start=0.3,
            )
        else:
            scheduler = ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.1, patience=10
            )

        return scheduler

    def main(self):
        log.info("Starting training...")
        
        # if self.config.gd == 0:
        #     self.input_df = self.input_df[~self.input_df["scan"].str.contains("Gd")]
        
        # Log GPU memory info at start
        if self.use_cuda:
            log.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
            log.info(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        self.output_df["prediction"] = np.nan
        
        # Check if k-fold cross-validation should be used
        if self.config.use_kfold:
            log.info(f"Using k-fold cross-validation with k={self.config.k_folds}")
            fold_results = self.k_fold_cross_validation()
            self._log_kfold_results(fold_results)
            return

        if self.config.use_train_validation_cols:
            training_rows = self.input_df.loc[self.input_df["training"] == 1]
            train_subjects = list(training_rows["subject_id"])
            validation_rows = self.input_df.loc[self.input_df["validation"] == 1]
            val_subjects = list(validation_rows["subject_id"])
        else:
            train_subjects, val_subjects = self.split_train_validation()

        # Store val_subjects as instance variable to use later
        self.val_subjects = val_subjects

        self.train_dl = self.data_handler.init_dl(self.folder, train_subjects)
        val_dl = self.data_handler.init_dl(self.folder, val_subjects, is_val_set=True)

        # Add scheduler initialization
        self.scheduler = self._init_scheduler(self.train_dl)

        loop_handler = TrainingLoop(
            self.model_handler,
            self.optimizer,
            self.device,
            self.input_df,
            self.config,
            self.scheduler,
        )

        best_val_loss = float("inf")

        for epoch in range(1, self.config.epochs + 1):
            log.info(f"Epoch {epoch}/{self.config.epochs}")
            
            # Log GPU memory before epoch
            if self.use_cuda:
                log.info(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                log.info(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

            trn_metrics = loop_handler.train_epoch(epoch, self.train_dl)
            val_metrics = loop_handler.validate_epoch(epoch, val_dl)

            self.tb_logger.log_metrics(
                "trn", epoch, trn_metrics, loop_handler.total_samples
            )
            self.tb_logger.log_metrics(
                "val", epoch, val_metrics, loop_handler.total_samples
            )

            # Calculate validation loss
            val_loss = val_metrics[METRICS_LOSS_NDX].mean().item()

            # Step the scheduler based on its type
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            elif isinstance(self.scheduler, OneCycleLR):
                # OneCycleLR is stepped in the training loop after each batch
                # Don't step it here
                pass
            else:
                # For StepLR, CosineAnnealingLR
                self.scheduler.step()

            # Track best model (optional)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                log.info(f"New best validation loss: {best_val_loss}")
                # Save the best model here
                self.model_handler.save_model(self.config.model_save_location)

            # Log current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            log.info(f"Current learning rate: {current_lr}")
            
            if self.use_cuda:
                torch.cuda.empty_cache()

        input_csv_location = self.config.csv_input_file
        subjects, sessions, runs, suffixes, actual_scores, predict_vals = get_validation_info(
            self.config.model,
            self.config.model_save_location,
            input_csv_location,
            self.val_subjects,
            self.config.folder,
        )

        output_csv_location = self.config.csv_output_file
        output_df = add_predicted_values(
            subjects, sessions, runs, suffixes, predict_vals, input_csv_location
        )
        output_csv_folder_name = get_folder_name(output_csv_location)
        if not os.path.exists(output_csv_folder_name):
            os.makedirs(output_csv_folder_name)
        output_df.to_csv(output_csv_location, index=False)
        standardized_rmse = compute_standardized_rmse(actual_scores, predict_vals)
        log.info(f"standardized_rmse: {standardized_rmse}")
        create_scatter_plot(actual_scores, predict_vals, self.config.plot_location)
        correlation_coefficient = create_correlation_coefficient(
            actual_scores, predict_vals
        )
        log.info(f"correlation_coefficient: {correlation_coefficient}")

        _, p_value = stats.pearsonr(actual_scores, predict_vals)
        log.info(f"Pearson correlation p-value: {p_value}")

        _, p_value = stats.spearmanr(actual_scores, predict_vals)
        log.info(f"Spearman correlation p-value: {p_value}")

    def split_train_validation(self):
        """
        Split subjects into training and validation sets at runtime with multiple strategies.
        If training/validation columns exist, use them.
        Otherwise, create a split based on the specified strategy.
        """
        # Check if training and validation columns exist and have valid data
        if (
            "training" in self.input_df.columns
            and "validation" in self.input_df.columns
            and self.input_df["training"].notna().any()
            and self.input_df["validation"].notna().any()
        ):
            # Use existing columns
            training_rows = self.input_df.loc[self.input_df["training"] == 1]
            validation_rows = self.input_df.loc[self.input_df["validation"] == 1]
            validation_users = list(
                set(validation_rows["subject_id"].to_list())
            )
            training_users = list(set(training_rows["subject_id"].to_list()))

            log.info(
                f"Using existing train/validation columns: {len(training_users)} train, {len(validation_users)} validation subjects"
            )
            return training_users, validation_users

        else:
            # Create train/validation split at runtime
            log.info(
                f"Train/validation columns not found or empty. Creating runtime {self.config.split_strategy} split..."
            )

            # Set random seed for reproducibility
            np.random.seed(self.config.random_seed)

            # Get unique subjects
            all_subjects = list(set(self.input_df["subject_id"].to_list()))
            n_subjects = len(all_subjects)

            if self.config.split_strategy == "random":
                training_users, validation_users = self._random_split(all_subjects)

            elif self.config.split_strategy == "stratified":
                training_users, validation_users = self._stratified_split(all_subjects)

            elif self.config.split_strategy == "sequential":
                training_users, validation_users = self._sequential_split(all_subjects)

            # Create the columns in the dataframe for consistency and future reference
            self.input_df["training"] = self.input_df["subject_id"].apply(
                lambda x: 1 if x in training_users else 0
            )
            self.input_df["validation"] = self.input_df["subject_id"].apply(
                lambda x: 1 if x in validation_users else 0
            )

            log.info(
                f"Created {self.config.split_strategy} split: {len(training_users)} train, {len(validation_users)} validation subjects"
            )
            log.info(f"Train split ratio: {len(training_users) / n_subjects:.2f}")
            log.info(
                f"Train subjects: {training_users[:3]}..."
                if len(training_users) > 3
                else f"Train subjects: {training_users}"
            )
            log.info(
                f"Validation subjects: {validation_users[:3]}..."
                if len(validation_users) > 3
                else f"Validation subjects: {validation_users}"
            )

            return training_users, validation_users

    def _random_split(self, all_subjects):
        """Random split of subjects"""
        shuffled_subjects = np.random.permutation(all_subjects)
        train_split = int(self.config.train_split * len(all_subjects))

        training_users = shuffled_subjects[:train_split].tolist()
        validation_users = shuffled_subjects[train_split:].tolist()

        return training_users, validation_users

    def _stratified_split(self, all_subjects):
        """Stratified split based on QU_motion score distribution"""
        try:
            # Get average QU_motion score per subject
            mean_values = self.input_df.groupby("subject_id")[
                "QU_motion"
            ].mean()
            sorted_means = mean_values.sort_values()
            i = 0
            training_users = []
            validation_users = []

            for index_label, _ in sorted_means.items():
                if i % 5 == 0:
                    validation_users.append(index_label)
                else:
                    training_users.append(index_label)
                i += 1

            log.info("Used stratified split based on QU_motion score distribution")
            return training_users, validation_users

        except Exception as e:
            log.warning(f"Stratified split failed ({e}), falling back to random split")
            return self._random_split(all_subjects)

    def _sequential_split(self, all_subjects):
        """Sequential split (first N subjects for training)"""
        # Sort subjects for reproducible sequential split
        sorted_subjects = sorted(all_subjects)
        train_split = int(self.config.train_split * len(sorted_subjects))

        training_users = sorted_subjects[:train_split]
        validation_users = sorted_subjects[train_split:]

        return training_users, validation_users

    def k_fold_cross_validation(self):
        """
        Implement stratified k-fold cross-validation for robust performance estimation.
        Stratifies folds based on QU_motion score distribution to ensure balanced score
        representation across all folds.
        Each fold trains a separate model and evaluates on its validation set.
        Returns metrics for each fold individually.
        """
        all_subjects = list(set(self.input_df["subject_id"].to_list()))
        k = self.config.k_folds
        
        log.info(f"Starting stratified {k}-fold cross-validation with {len(all_subjects)} unique subjects")
        
        # Calculate average QU_motion per subject for stratification
        subject_motion_map = {}
        for subject in all_subjects:
            subject_rows = self.input_df[self.input_df["subject_id"] == subject]
            avg_motion = subject_rows["QU_motion"].mean()
            subject_motion_map[subject] = avg_motion
        
        # Create stratification labels by binning QU_motion scores
        # Using percentile-based binning to create balanced strata
        motion_values = np.array([subject_motion_map[s] for s in all_subjects])
        try:
            # Create k strata based on percentiles
            strata = pd.qcut(motion_values, q=k, labels=False, duplicates='drop')
            log.info(f"Created {len(np.unique(strata))} strata based on QU_motion percentiles")
        except Exception as e:
            log.warning(f"Percentile-based stratification failed ({e}), falling back to random k-fold")
            strata = np.random.randint(0, k, len(all_subjects))
        
        skfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=self.config.random_seed)
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(skfold.split(all_subjects, strata)):
            log.info(f"\n{'='*60}")
            log.info(f"Training fold {fold + 1}/{k}")
            log.info(f"{'='*60}")
            
            train_subjects = [all_subjects[i] for i in train_idx]
            val_subjects = [all_subjects[i] for i in val_idx]
            
            # Log QU_motion distribution for this fold
            train_motion_scores = [subject_motion_map[s] for s in train_subjects]
            val_motion_scores = [subject_motion_map[s] for s in val_subjects]
            log.info(f"Fold {fold + 1}: {len(train_subjects)} training subjects, {len(val_subjects)} validation subjects")
            log.info(f"  Training QU_motion - Mean: {np.mean(train_motion_scores):.3f}, Std: {np.std(train_motion_scores):.3f}")
            log.info(f"  Validation QU_motion - Mean: {np.mean(val_motion_scores):.3f}, Std: {np.std(val_motion_scores):.3f}")
            
            # Train model for this fold
            fold_metrics = self.train_fold(fold, train_subjects, val_subjects)
            fold_results.append(fold_metrics)
        
        return fold_results
    
    def train_fold(self, fold_num, train_subjects, val_subjects):
        """
        Train a single fold and return metrics.
        
        Args:
            fold_num (int): Fold number (0-indexed)
            train_subjects (list): List of training subject IDs
            val_subjects (list): List of validation subject IDs
            
        Returns:
            dict: Dictionary containing fold metrics (validation loss, RMSE, correlation, etc.)
        """
        # Create a fold-specific dataframe with 'training' and 'validation' columns
        # This is required by TrainingLoop for computing weighted loss
        fold_df = self.input_df.copy()
        fold_df['training'] = fold_df['subject_id'].apply(lambda x: 1 if x in train_subjects else 0)
        fold_df['validation'] = fold_df['subject_id'].apply(lambda x: 1 if x in val_subjects else 0)
        
        # Create new model handler for this fold
        fold_model_handler = ModelHandler(self.config.model, self.use_cuda, self.device)
        fold_optimizer = self._init_optimizer()
        
        # Create dataloaders for this fold
        train_dl = self.data_handler.init_dl(self.folder, train_subjects, is_val_set=False)
        val_dl = self.data_handler.init_dl(self.folder, val_subjects, is_val_set=True)
        
        # Initialize scheduler
        scheduler = self._init_scheduler(train_dl)
        
        # Create fold-specific model save location
        base_path = self.config.model_save_location
        if base_path.endswith('.pt'):
            fold_model_path = base_path[:-3] + f"_fold_{fold_num}.pt"
        else:
            fold_model_path = base_path + f"_fold_{fold_num}.pt"
        
        # Create training loop for this fold, passing the fold-specific dataframe
        loop_handler = TrainingLoop(
            fold_model_handler,
            fold_optimizer,
            self.device,
            fold_df,
            self.config,
            scheduler,
        )
        
        best_val_loss = float("inf")
        fold_metrics = {
            'fold': fold_num,
            'best_val_loss': None,
            'final_standardized_rmse': None,
            'final_correlation_coefficient': None,
            'final_pearson_pvalue': None,
            'final_spearman_pvalue': None,
            'train_subjects': train_subjects,
            'val_subjects': val_subjects,
        }
        
        # Training loop for this fold
        for epoch in range(1, self.config.epochs + 1):
            log.info(f"Fold {fold_num + 1} - Epoch {epoch}/{self.config.epochs}")
            
            trn_metrics = loop_handler.train_epoch(epoch, train_dl)
            val_metrics = loop_handler.validate_epoch(epoch, val_dl)
            
            # Calculate validation loss
            val_loss = val_metrics[METRICS_LOSS_NDX].mean().item()
            
            # Step the scheduler
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            elif isinstance(scheduler, OneCycleLR):
                pass
            else:
                scheduler.step()
            
            # Track best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                log.info(f"Fold {fold_num + 1} - New best validation loss: {best_val_loss}")
                fold_metrics['best_val_loss'] = best_val_loss
                # Save the best model for this fold using fold-specific path
                fold_model_handler.save_model(fold_model_path)
        
        # Compute final metrics on validation set using fold-specific model
        subjects, sessions, runs, suffixes, actual_scores, predict_vals = get_validation_info(
            self.config.model,
            fold_model_path,
            self.config.csv_input_file,
            val_subjects,
            self.config.folder,
        )
        
        standardized_rmse = compute_standardized_rmse(actual_scores, predict_vals)
        correlation_coefficient = create_correlation_coefficient(actual_scores, predict_vals)
        pearson_r, pearson_p = stats.pearsonr(actual_scores, predict_vals)
        spearman_r, spearman_p = stats.spearmanr(actual_scores, predict_vals)
        
        fold_metrics['final_standardized_rmse'] = standardized_rmse
        fold_metrics['final_correlation_coefficient'] = correlation_coefficient
        fold_metrics['final_pearson_pvalue'] = pearson_p
        fold_metrics['final_spearman_pvalue'] = spearman_p
        
        log.info(f"Fold {fold_num + 1} - Standardized RMSE: {standardized_rmse}")
        log.info(f"Fold {fold_num + 1} - Correlation Coefficient: {correlation_coefficient}")
        log.info(f"Fold {fold_num + 1} - Pearson p-value: {pearson_p}")
        log.info(f"Fold {fold_num + 1} - Spearman p-value: {spearman_p}")
        
        if self.use_cuda:
            torch.cuda.empty_cache()
        
        return fold_metrics
    
    def _log_kfold_results(self, fold_results):
        """
        Log and summarize k-fold cross-validation results across all folds.
        
        Args:
            fold_results (list): List of dictionaries containing metrics for each fold
        """
        log.info(f"\n{'='*60}")
        log.info("K-FOLD CROSS-VALIDATION SUMMARY")
        log.info(f"{'='*60}")
        
        rmse_values = []
        corr_values = []
        pearson_p_values = []
        spearman_p_values = []
        
        for fold_result in fold_results:
            fold_num = fold_result['fold']
            rmse = fold_result['final_standardized_rmse']
            corr = fold_result['final_correlation_coefficient']
            pearson_p = fold_result['final_pearson_pvalue']
            spearman_p = fold_result['final_spearman_pvalue']
            
            log.info(f"\nFold {fold_num + 1}:")
            log.info(f"  Standardized RMSE: {rmse:.6f}")
            log.info(f"  Correlation: {corr:.6f}")
            log.info(f"  Pearson p-value: {pearson_p:.6e}")
            log.info(f"  Spearman p-value: {spearman_p:.6e}")
            
            if rmse is not None:
                rmse_values.append(rmse)
            if corr is not None:
                corr_values.append(corr)
            if pearson_p is not None:
                pearson_p_values.append(pearson_p)
            if spearman_p is not None:
                spearman_p_values.append(spearman_p)
        
        # Compute statistics across folds
        log.info(f"\n{'='*60}")
        log.info("AGGREGATED METRICS ACROSS ALL FOLDS")
        log.info(f"{'='*60}")
        
        if rmse_values:
            log.info(f"Standardized RMSE - Mean: {np.mean(rmse_values):.6f}, Std: {np.std(rmse_values):.6f}")
        if corr_values:
            log.info(f"Correlation - Mean: {np.mean(corr_values):.6f}, Std: {np.std(corr_values):.6f}")
        if pearson_p_values:
            log.info(f"Pearson p-value - Mean: {np.mean(pearson_p_values):.6e}, Std: {np.std(pearson_p_values):.6e}")
        if spearman_p_values:
            log.info(f"Spearman p-value - Mean: {np.mean(spearman_p_values):.6e}, Std: {np.std(spearman_p_values):.6e}")
        
        # Save fold results to CSV for detailed analysis
        fold_results_df = pd.DataFrame(fold_results)
        results_file = os.path.join(self.tb_run_dir, f"kfold_results_{self.time_str}.csv")
        fold_results_df.to_csv(results_file, index=False)
        log.info(f"\nDetailed fold results saved to: {results_file}")


def main():
    autoQcTrainingApp = AutoQcTrainingApp(sys_argv=sys.argv)
    autoQcTrainingApp.main()


if __name__ == "__main__":
    main()
