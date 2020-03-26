import time
from typing import Dict, Any, List
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils import data
from torch.optim.lr_scheduler import ReduceLROnPlateau

from baselines.models_factory import ModelsFactory
from baselines.datasets_factory import DatasetsFactory
from baselines.tracking_utils import ResultsAnalyzer
from baselines.supported_models import DOUBLE_OUTPUT_MODELS, NO_LABELS_MODELS


def save_checkpoint(model: nn.Module, model_name: str, dev_iou: float, checkpoint_dir: str) -> None:
    current_date = date.today().strftime("%d-%m-%y")

    # create checkpoint folder if it doesn't exist
    checkpoint_path = Path(checkpoint_dir) / model_name
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # save the model to dict
    checkpoint_file = checkpoint_path / f"{current_date}_{dev_iou}.pth"
    torch.save(model.state_dict(), checkpoint_file)
    print(f"Saved best model so far on dev set with type {model_name} and performance mean IoU of: {dev_iou}")


def inference_and_iou_comp(model_name: str, model: nn.Module, compute_device: torch.device, data_loader: data.DataLoader, dataset_length: int,
                           reg_loss_function: nn.Module) -> Any:
    model.eval()
    model.to(compute_device)

    with torch.no_grad():

        dataset_indices: List[str] = []
        dataset_predictions: List[np.ndarray] = []
        dataset_labels: List[np.ndarray] = []
        dataset_containment: List[np.ndarray] = []
        total_loss = 0
        current_sample_idx = 0
        frame_shapes = np.array([320, 240, 320, 240])

        for batch_idx, sample in enumerate(data_loader):
            x, y, _ = sample
            boxes, _ = x
            labels, mask = y
            current_batch_size = len(labels)

            boxes = boxes.to(compute_device)
            labels, mask = labels.to(compute_device), mask.to(compute_device)

            if model_name in DOUBLE_OUTPUT_MODELS:
                output, index_to_track_prediction = model(boxes)

            else:
                output = model(boxes)

            # prediction loss
            pred_loss = reg_loss_function(output, labels)

            # consistency loss
            next_output_frames = output[:, 1:, :]
            current_output_frames = output[:, :-1, :]
            consistency_loss = torch.mean(torch.norm(next_output_frames - current_output_frames, p=2, dim=-1))

            if model_name in NO_LABELS_MODELS:
                pred_loss = pred_loss * mask  # mask contains only visible objects
                pred_loss = torch.mean(pred_loss)

            else:
                pred_loss = torch.mean(pred_loss)

            if model_name in NO_LABELS_MODELS:
                loss = pred_loss + 0.5 * consistency_loss

            else:
                loss = pred_loss

            # move outputs to cpu and flatten output and labels
            batch_indices = [str(i + current_sample_idx) for i in range(current_batch_size)]
            batch_predictions = output.cpu().numpy().reshape(-1, 4)
            batch_labels = labels.cpu().numpy().reshape(-1, 4)

            dataset_indices.extend(batch_indices)
            dataset_predictions.extend(batch_predictions)
            dataset_labels.extend(batch_labels)
            dataset_containment.extend(torch.sum(mask, dim=-1).type(torch.bool).cpu().numpy())

            current_sample_idx += current_batch_size
            total_loss += loss.item() * current_batch_size

        # transform back to pixels scale (undo normalization)
        snitch_predictions: np.ndarray = (np.array(dataset_predictions) * frame_shapes).reshape(
            (dataset_length, 300, 4)).astype(np.int32)
        snitch_labels: np.ndarray = (np.array(dataset_labels) * frame_shapes).reshape(
            (dataset_length, 300, 4)).astype(np.int32)
        containment_mask: Dict[str, np.ndarray] = {str(i): dataset_containment[i] for i in range(dataset_length)}

        # analyze performance
        average_loss = total_loss / len(dataset_indices)
        analyzer = ResultsAnalyzer(dataset_indices, snitch_predictions, snitch_labels)
        analyzer.compute_aggregated_metric("video_mean", np.mean)
        analyzer.compute_aggregated_metric_masking_frames("containment", np.mean, containment_mask)

        iou_results: pd.DataFrame = analyzer.get_analysis_df()
        try:
            mean_iou = np.mean(iou_results["video_mean_iou"]).item()
            containment_mean_iou = np.mean(iou_results["containment_mean_iou"]).item()
        except AttributeError:
            mean_iou = 0.0
            containment_mean_iou = 0.0

        return average_loss, mean_iou, containment_mean_iou


def training_main(model_name: str, train_config: Dict[str, Any], model_config: Dict[str, int]):

    # create train and dev datasets using the files specified in the training configuration
    train_samples_dir = train_config["train_sample_dir"]
    train_labels_dir = train_config["train_labels_dir"]
    train_containment_file = train_config["train_containment_file"]

    dev_samples_dir = train_config["dev_sample_dir"]
    dev_labels_dir = train_config["dev_labels_dir"]
    dev_containment_file = train_config["dev_containment_file"]

    train_dataset: data.Dataset = DatasetsFactory.get_training_dataset(model_name, train_samples_dir, train_labels_dir, train_containment_file)
    dev_dataset: data.Dataset = DatasetsFactory.get_training_dataset(model_name, dev_samples_dir, dev_labels_dir, dev_containment_file)

    # training hyper parameters and configuration
    batch_size = train_config["batch_size"]
    num_workers = train_config["num_workers"]
    num_epochs = train_config["num_epochs"]
    learning_rate = train_config["learning_rate"]
    print_batch_step = train_config["print_step"]
    inference_batch_size = train_config["inference_batch_size"]
    scheduler_patience = train_config["lr_scheduler_patience"]
    scheduler_factor = train_config["lr_scheduler_factor"]
    checkpoints_path = train_config["checkpoints_path"]
    device = torch.device(train_config["device"])
    # consistency_rate = train_config["consistency_rate"]

    # model, loss and optimizer
    model: nn.Module = ModelsFactory.get_model(model_name, model_config)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, verbose=True)
    loss_function = nn.L1Loss(reduction="none")

    # create data loaders
    train_config_dict = {"batch_size": batch_size, "num_workers": num_workers}
    inference_config_dict = {"batch_size": inference_batch_size, "num_workers": num_workers}
    training_loader = data.DataLoader(train_dataset, **train_config_dict)
    train_inference_loader = data.DataLoader(train_dataset, **inference_config_dict)
    dev_loader = data.DataLoader(dev_dataset, **inference_config_dict)

    # Start training
    model = model.to(device)
    highest_dev_iou: float = 0
    train_start_time = time.time()

    for epoch in range(num_epochs):
        model.train(mode=True)
        epoch_num = epoch + 1

        # loss statistics
        batches_running_loss = 0
        batches_running_pred_loss = 0
        batches_running_const_loss = 0

        for batch_idx, sample in enumerate(training_loader, 1):

            x, y, _ = sample
            boxes, _ = x
            labels, mask = y
            boxes = boxes.to(device)
            labels, mask = labels.to(device), mask.to(device)

            optimizer.zero_grad()

            if model_name in DOUBLE_OUTPUT_MODELS:
                output, index_to_track_prediction = model(boxes)

            else:
                output = model(boxes)

            # prediction loss
            pred_loss = loss_function(output, labels)

            # consistency loss
            next_output_frames = output[:, 1:, :]
            current_output_frames = output[:, :-1, :]
            consistency_loss = torch.mean(torch.norm(next_output_frames - current_output_frames, p=2, dim=-1))

            if model_name in NO_LABELS_MODELS:
                pred_loss = pred_loss * mask  # mask contains only visible objects
                pred_loss = torch.mean(pred_loss)

            else:
                pred_loss = torch.mean(pred_loss)

            if model_name in NO_LABELS_MODELS:
                loss = pred_loss + 0.5 * consistency_loss

            else:
                loss = pred_loss

            batches_running_loss += loss.item()
            batches_running_pred_loss += pred_loss.item()
            batches_running_const_loss += consistency_loss.item()

            loss.backward()
            optimizer.step()

            # print inter epoch statistics
            if batch_idx % print_batch_step == 0:

                num_samples_seen = batch_idx * batch_size
                num_samples_total = len(train_dataset)
                epoch_complete_ratio = 100 * batch_idx / len(training_loader)
                average_running_loss = batches_running_loss / print_batch_step
                average_pred_loss = batches_running_pred_loss / print_batch_step
                average_consist_loss = batches_running_const_loss / print_batch_step
                time_since_beginning = int(time.time() - train_start_time)

                print("Train Epoch: {} [{}/{} ({:.0f}%)]\t Average Loss: Total {:.4f}, Pred {:.4f} Consistent {:.4f} Training began {} seconds ago".format(
                    epoch_num, num_samples_seen, num_samples_total, epoch_complete_ratio, average_running_loss,
                    average_pred_loss, average_consist_loss, time_since_beginning
                ))

                batches_running_loss = 0
                batches_running_pred_loss = 0
                batches_running_const_loss = 0

        # end of epoch - compute mean iou over train and dev
        train_loss, train_miou, train_containment_miou = inference_and_iou_comp(model_name, model, device, train_inference_loader, len(train_dataset), loss_function)
        dev_loss, dev_miou, dev_containment_miou = inference_and_iou_comp(model_name, model, device, dev_loader, len(dev_dataset), loss_function)

        print("Epoch {} Training Set: Loss {:.4f}, Mean IoU {:.6f}, Mask Mean Iou {:.6f}".format(epoch_num, train_loss, train_miou, train_containment_miou))
        print("Epoch {} Dev Set: Loss {:.4f}, Mean IoU {:.6f}, Mask Mean Iou {:.6f}".format(epoch_num, dev_loss, dev_miou, dev_containment_miou))

        # learning rate scheduling
        scheduler.step(train_loss)

        # check if it is the best performing model so far and save it
        if dev_miou > highest_dev_iou:
            highest_dev_iou = dev_miou
            save_checkpoint(model, model_name, round(highest_dev_iou, 3), checkpoints_path)
