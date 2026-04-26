"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb
from .datasets.road_dataset import load_data
from .models import save_model, load_model, MLPPlanner
from .metrics import PlannerMetric

import warnings
from torch.jit import TracerWarning

warnings.filterwarnings("ignore", category=TracerWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def train(
    exp_dir: str = "logs",
    model_name: str = "mlp_planner",
    n_track: int = 10,
    n_waypoints: int = 3,
    num_epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # load model from factory 
    model = load_model(model_name, n_track=n_track, n_waypoints=n_waypoints)
    model = model.to(device)

    # load data
    is_cnn = model_name == "cnn_planner"
    train_loader = load_data("drive_data/train", transform_pipeline="aug", batch_size=batch_size, shuffle=True)
    val_loader = load_data("drive_data/val", transform_pipeline="default" if is_cnn else "state_only", batch_size=batch_size, shuffle=False)

    # create loss function and optimizer
    loss_func = torch.nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    global_step = 0
    train_metrics = PlannerMetric()
    val_metrics = PlannerMetric()

    # training loop
    for epoch in range(num_epochs):
        train_metrics.reset()
        val_metrics.reset()

        model.train()

        for batch in train_loader:
            waypoints = batch["waypoints"].to(device)
            waypoints_mask = batch["waypoints_mask"].to(device)

            optimizer.zero_grad()
            
            if is_cnn:
                output = model(image=batch["image"].to(device))
            else:
                output = model(
                    track_left=batch["track_left"].to(device),
                    track_right=batch["track_right"].to(device)
                )
            
            # Loss only on valid waypoints
            loss = loss_func(output[waypoints_mask], waypoints[waypoints_mask])
            loss.backward()
            optimizer.step()

            # Planner metric needs predictions, true labels, and mask
            train_metrics.add(output, waypoints, waypoints_mask)
            
            logger.add_scalar("train_loss", loss.item(), global_step)
            global_step += 1

        # disable gradient computation and switch to evaluation mode
        model.eval()
        with torch.inference_mode():
            for batch in val_loader:
                waypoints = batch["waypoints"].to(device)
                waypoints_mask = batch["waypoints_mask"].to(device)

                if is_cnn:
                    output = model(image=batch["image"].to(device))
                else:
                    output = model(
                        track_left=batch["track_left"].to(device),
                        track_right=batch["track_right"].to(device)
                    )
                val_metrics.add(output, waypoints, waypoints_mask)


        # log average train and val metrics to tensorboard
        train_res = train_metrics.compute()
        val_res = val_metrics.compute()
        epoch_train_l1 = train_res["l1_error"]
        epoch_val_l1 = val_res["l1_error"]

        logger.add_scalar("train_l1_error", epoch_train_l1, epoch)
        logger.add_scalar("val_l1_error", epoch_val_l1, epoch)

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epochs - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epochs:2d}: "
                f"train_l1={epoch_train_l1:.4f} "
                f"val_l1={epoch_val_l1:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

        # Update the learning rate based on validation loss
        scheduler.step(epoch_val_l1)

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, default="mlp_planner")
    parser.add_argument("--n_track", type=int, default=10)
    parser.add_argument("--n_waypoints", type=int, default=3)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--batch_size", type=int, default=64)  

    # pass all arguments to train
    train(**vars(parser.parse_args()))