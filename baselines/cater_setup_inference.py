import json
import numpy as np
from typing import List, Dict

import torch
import torch.nn as nn
import pandas as pd
from torch.utils import data

from baselines.models_factory import ModelsFactory
from baselines.datasets_factory import DatasetsFactory
from baselines.proj_utils import get_class_prediction


W_FRAME = 320
H_FRAME = 240


def transform_xyxy_to_w_h(predictions: np.ndarray) -> np.ndarray:
    w_h = [[(x2 + x1) / 2, (y2 + y1) / 2] for x1, y1, x2, y2 in predictions]
    return np.array(w_h)


def get_classes_predictions(predictions: np.ndarray) -> List[int]:
    class_predictions = []
    for pred in predictions:
        pred_w = pred[0]
        pred_h = pred[1]
        pred_class = get_class_prediction(pred_w * 2 / W_FRAME - 1, pred_h * 2 / H_FRAME - 1, nrows=3, ncols=3)
        class_predictions.append(pred_class)

    return class_predictions


def cater_setup_inference(model_name: str, results_dir: str, inference_config_path: str, model_config_path: str):
    with open(inference_config_path, "rb") as f:
        config: Dict[str, str] = json.load(f)

    with open(model_config_path, "rb") as f:
        model_config: Dict[str, int] = json.load(f)

    samples_dir = config["sample_dir"]
    labels_dir = config["labels_dir"]
    batch_size = int(config["batch_size"])
    num_workers = int(config["num_workers"])
    model_path = config["model_path"]
    device = torch.device(config["device"])

    dataset: data.Dataset = DatasetsFactory.get_inference_dataset(model_name, samples_dir, labels_dir)
    data_loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    dataset_length = len(dataset)

    # load model
    model: nn.Module = ModelsFactory.get_model(model_name, model_config, model_path)

    # predict model results
    dataset_videos_indices: Dict[str, int] = {}
    dataset_predictions: List[np.ndarray] = []
    current_sample_idx = 0

    model.eval()
    model.to(device)
    with torch.no_grad():
        frame_shapes = np.array([320, 240, 320, 240])

        for batch_idx, sample in enumerate(data_loader):

            x, y, video_names = sample
            boxes, index_to_track_labels = x
            labels, _ = y
            current_batch_size = len(labels)

            boxes = boxes.to(device)

            output, _ = model(boxes)

            # output only prediction in for last frame
            output = output[:, -1, :]

            # move outputs to cpu and flatten output and labels
            batch_videos = {video_names[i]: i + current_sample_idx for i in range(current_batch_size)}
            batch_predictions = output.cpu().numpy().reshape(-1, 4)

            dataset_videos_indices.update(batch_videos)
            dataset_predictions.extend(batch_predictions)
            current_sample_idx += current_batch_size

    dataset_predictions = (np.array(dataset_predictions) * frame_shapes).reshape((dataset_length, 4)).astype(np.int32)
    cx_cy_output = transform_xyxy_to_w_h(dataset_predictions)
    pred_classes = get_classes_predictions(cx_cy_output)

    results = {
        "video_names": [],
        "class_predictions": []
    }

    for video_name, video_index in dataset_videos_indices.items():
        results["video_names"].append(f"{video_name}.avi")
        results["class_predictions"].append(pred_classes[video_index])

    results_df = pd.DataFrame(results)
    results_file = f"{results_dir}/class_pred_results.csv"
    results_df.to_csv(results_file, index=False)


