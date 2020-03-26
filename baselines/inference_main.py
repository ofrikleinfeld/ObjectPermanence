import json
import numpy as np
import pickle
from typing import List, Dict
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils import data

from baselines.tracking_utils import VideoHandling, DataHelper
from baselines.models_factory import ModelsFactory, AbstractReasoner
from baselines.datasets_factory import DatasetsFactory
from baselines.supported_models import DOUBLE_OUTPUT_MODELS


LARGE_CONS_INDICES = list(range(0, 64, 4))  # according to class -> index mapping file
SNITCH_NAME = "small_gold_spl_metal_Spl_0"


def get_experiment_videos(config: Dict[str, str]) -> List[str]:
    videos_dir = config["videos_dir"]

    if "sample_file" not in config:
        return [str(vid_path) for vid_path in Path(videos_dir).glob("*.avi")]

    else:
        sample_file_path = config["sample_file"]
        all_video_paths = list(Path(videos_dir).glob("*.avi"))
        all_videos_names: Dict[str, str] = {path.stem: path for path in all_video_paths}
        select_videos_paths: List[str] = []

        with open(sample_file_path, "r") as sample_file:
            for line in sample_file:
                video_path = line[:-1]
                video_name = Path(video_path).stem
                video_path = all_videos_names[video_name]
                select_videos_paths.append(video_path)

        return select_videos_paths


def track_and_predict(video_name: str, video_path: Path, model: AbstractReasoner,
                      predictions_path: Path, labels_path: Path, results_dir: str) -> List[List[int]]:

    out_vid_path = Path(results_dir) / (video_name + "_results.avi")
    snitch_bb_predictions: List[List[int]] = []

    # load predictions and also labels (used for debugging videos)
    with open(str(predictions_path), 'rb') as f:
        prediction_data = pickle.load(f)

    with open(str(labels_path), "rb") as f:
        video_labels: Dict[str, List[List[int]]] = json.load(f)

    snitch_labels = video_labels[SNITCH_NAME]

    # convert from x,y,w,h to x,y,x,y
    snitch_labels = [[x, y, x + w, y + h] for x, y, w, h in snitch_labels]

    # start reading video, tracking and writing to output video file
    # transform to str from Path object (cv2 cannot handle these objects)
    vid_path = str(video_path)
    out_vid_path = str(out_vid_path)
    video_handler = VideoHandling(vid_path, out_vid_path)

    # start reading video frames and predict
    video_handler.read_next_frame()
    frame = video_handler.get_current_frame()
    current_frame_index = video_handler.get_current_frame_index()
    video_still_active = video_handler.check_video_still_active()

    while video_still_active:

        # track
        model.track_for_frame(frame, current_frame_index, prediction_data, video_name)
        state = model.state
        snitch_visible = model.snitch_visible

        # Draw the current tracked object BB
        if snitch_visible:
            current_tracked_bb = state["snitch_box"]
        else:
            cx, cy = state['target_pos']
            w_box, h_box = state['target_sz']

            current_tracked_bb = [int(cx - w_box / 2), int(cy - h_box / 2), int(cx + w_box / 2), int(cy + h_box / 2)]

        video_handler.write_bb_to_frame(current_tracked_bb, color=(0, 255, 255))

        # Drawing the GT rectangle as well
        current_labels = snitch_labels[current_frame_index]
        video_handler.write_bb_to_frame(current_labels, color=(255, 0, 0))

        # Draw the location of the global object to track
        # (if the model supports it)

        if "object_sz" in state and not snitch_visible:
            object_w_box, object_h_box = state['object_sz']
            object_cx = cx
            object_cy = cy

            current_tracked_object_index = model.state["object_label"]
            if current_tracked_object_index in LARGE_CONS_INDICES:
                object_cy += 15

            current_pred_bb = [int(object_cx - object_w_box / 2), int(object_cy - object_h_box / 2),
                               int(object_cx + object_w_box / 2), int(object_cy + object_h_box / 2)]

            video_handler.write_bb_to_frame(current_pred_bb, color=(0, 0, 255))

        # write debug frame
        video_handler.write_debug_frame()

        # update current snitch_bb prediction according to model
        if "object_sz" in state and not snitch_visible:
            snitch_bb_prediction = current_pred_bb
        else:
            snitch_bb_prediction = current_tracked_bb

        snitch_bb_predictions.append(snitch_bb_prediction)

        # read the next frame
        video_handler.read_next_frame()
        frame = video_handler.get_current_frame()
        video_still_active = video_handler.check_video_still_active()
        current_frame_index = video_handler.get_current_frame_index()

    # complete video writing and end the video prediction
    video_handler.complete_video_writing()
    return snitch_bb_predictions


def trackers_inference_main(model_type: str, results_dir: str, config_path: str) -> None:

    # load configuration dict
    with open(config_path, "rb") as f:
        config: Dict[str, str] = json.load(f)

    # extract paths to video files for the experiment
    experiment_videos = get_experiment_videos(config)
    experiment_video_names = {str(Path(vid_path).stem): str(vid_path) for vid_path in experiment_videos}

    # define global parameters
    samples_dir = config["sample_dir"]
    labels_dir = config["labels_dir"]
    device = torch.device(config["device"]) if "device" in config else ""
    model_weights = config["model_weights"] if "model_weights" in config else ""

    # load a tracking model
    reasoner: AbstractReasoner = ModelsFactory.get_tracker_model(model_type, model_weights, device)

    for video_name, video_path in tqdm(experiment_video_names.items()):
        predictions_path = Path(samples_dir) / (video_name + ".pkl")
        labels_path = Path(labels_dir) / (video_name + "_bb.json")

        snitch_bb_prediction = track_and_predict(video_name, video_path, reasoner, predictions_path, labels_path, results_dir)
        DataHelper.write_bb_predictions_to_file(video_path, results_dir, snitch_bb_prediction)


def reasoning_inference_main(model_name: str, results_dir: str, inference_config_path: str, model_config_path: str):
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
    dataset_labels: List[np.ndarray] = []
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

            if model_name in DOUBLE_OUTPUT_MODELS:
                output, index_to_track_prediction = model(boxes)

            else:
                output = model(boxes)

            # move outputs to cpu and flatten output and labels
            batch_videos = {video_names[i]: i + current_sample_idx for i in range(current_batch_size)}
            batch_predictions = output.cpu().numpy().reshape(-1, 4)
            batch_labels = labels.numpy().reshape(-1, 4)

            dataset_videos_indices.update(batch_videos)
            dataset_predictions.extend(batch_predictions)
            dataset_labels.extend(batch_labels)
            current_sample_idx += current_batch_size

    dataset_predictions = (np.array(dataset_predictions) * frame_shapes).reshape((dataset_length, 300, 4)).astype(np.int32)
    dataset_labels = (np.array(dataset_labels) * frame_shapes).reshape((dataset_length, 300, 4)).astype(np.int32)

    # extract paths to video files for the experiment
    experiment_videos = get_experiment_videos(config)
    experiment_video_names = {str(Path(vid_path).stem): str(vid_path) for vid_path in experiment_videos}

    # write debug videos
    for video_name, video_path in tqdm(experiment_video_names.items()):
        out_vid_path = str(Path(results_dir) / (video_name + "_results.avi"))
        video_idx = dataset_videos_indices.get(video_name, None)

        if video_idx is not None:
            video_predictions = dataset_predictions[video_idx]
            video_labels = dataset_labels[video_idx]

            video_handler = VideoHandling(video_path, out_vid_path)

            # start reading video frames and predict
            video_handler.read_next_frame()
            video_still_active = video_handler.check_video_still_active()

            while video_still_active:
                current_frame_index = video_handler.get_current_frame_index()
                frame_pred = video_predictions[current_frame_index]
                frame_gt = video_labels[current_frame_index]

                video_handler.write_bb_to_frame(list(frame_pred), color=(0, 255, 255))
                video_handler.write_bb_to_frame(list(frame_gt), color=(255, 0, 0))
                video_handler.write_debug_frame()

                # read the next frame
                video_handler.read_next_frame()
                video_still_active = video_handler.check_video_still_active()

            video_handler.complete_video_writing()

            # write bb results to file for future offline analysis
            DataHelper.write_bb_predictions_to_file(video_path, results_dir, video_predictions)
