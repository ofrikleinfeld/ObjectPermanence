import pickle
import json
from typing import List, Tuple, Dict
from pathlib import Path
from tqdm import tqdm

import torch
import numpy as np

from baselines.inference_main import get_experiment_videos
from baselines.models_factory import ModelsFactory
from baselines.detector import CaterObjectDetector
from baselines.tracking_utils import VideoHandling


def output_video_predictions(video_path: str, detector: CaterObjectDetector, compute_device: torch.device) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    bb_predictions: List[np.ndarray] = []
    labels: List[np.ndarray] = []

    # start reading video, tracking and writing to output video file
    vid_path = str(video_path)
    video_handler = VideoHandling(vid_path)

    # start reading video frames
    video_handler.read_next_frame()
    video_still_active = video_handler.check_video_still_active()

    while video_still_active:
        # read next frame and predict bounding boxes for all present objects
        frame: np.ndarray = video_handler.get_current_frame()
        object_predictions = detector(frame, compute_device)
        object_predictions = detector.remove_low_probability_object(object_predictions[0])

        # update to predictions form the model output
        frame_bb_predictions: np.ndarray = object_predictions["boxes"].cpu().numpy().astype(np.int)
        frame_labels_predictions: np.ndarray = object_predictions["labels"].cpu().numpy().astype(np.int)
        bb_predictions.append(frame_bb_predictions)
        labels.append(frame_labels_predictions)

        # read the next frame
        video_handler.read_next_frame()
        video_still_active = video_handler.check_video_still_active()
    
    return bb_predictions, labels


def preprocess_video(process_args: List):

    # # load object detector
    # factory = ModelsFactory()
    # detector: CaterObjectDetector = factory.get_detector_model("object_detector", od_weights_path)
    # detector.load_model(device)
    #
    # # preform predictions for on the entire video and retrieve the results
    # bb_predictions, labels = output_video_predictions(video_path, detector, device)
    #
    # # save the predictions as pickle files
    # video_path = Path(video_path)
    # results_dir = Path(results_dir)
    # video_name = video_path.stem
    # output_path = results_dir / (video_name + ".pkl")
    # output_data = {"bb": bb_predictions, "labels": labels}
    # if (len(output_data["bb"]) == 300) and (len(output_data["labels"]) == 300):
    #     with open(output_path, "wb") as f:
    #         pickle.dump(output_data, f, pickle.HIGHEST_PROTOCOL)
    #
    #     print(f"Finished writing object detection outputs for video {video_name}")

    video_path, od_weights, results_dir = process_args
    print(video_path)
    # assign an available gpu for this process
    # num_possible_gpu_devices = torch.cuda.device_count()
    # assigned_gpu = multiprocessing.current_process().ident % num_possible_gpu_devices
    # assigned_gpu = multiprocessing.current_process().ident % 2 + 2
    # device = torch.device(f'cuda:{assigned_gpu}')
    device = torch.device(f'cuda:2')

    # load object detector
    factory = ModelsFactory()
    detector: CaterObjectDetector = factory.get_detector_model("object_detector", od_weights)
    detector.load_model(device)

    # preform predictions for on the entire video and retrieve the results
    bb_predictions, labels = output_video_predictions(video_path, detector, device)

    # save the predictions as pickle files
    video_path = Path(video_path)
    results_dir = Path(results_dir)
    video_name = video_path.stem
    output_path = results_dir / (video_name + ".pkl")
    output_data = {"bb": bb_predictions, "labels": labels}
    if (len(output_data["bb"]) == 300) and (len(output_data["labels"]) == 300):
        with open(output_path, "wb") as f:
            pickle.dump(output_data, f, pickle.HIGHEST_PROTOCOL)

        print(f"Finished writing object detection outputs for video {video_name}")


def preprocess_main(results_dir: str, config_path: str) -> None:
    with open(config_path, "rb") as f:
        config = json.load(f)

    # extract paths to video files for the experiment
    experiment_videos = get_experiment_videos(config)
    od_weights = config["od_model_weights"]
    num_videos = len(experiment_videos)

    process_args = [(video_path, od_weights, results_dir) for video_path in experiment_videos]
    for i in tqdm(range(num_videos)):
        try:
            process_args_i = process_args[i]
            preprocess_video(process_args_i)
        except Exception:
            continue
