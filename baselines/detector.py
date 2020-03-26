from typing import Dict, Tuple

import torch
import torch.nn as nn
import numpy as np
import cv2

from object_detection.models import get_fast_rcnn_for_fine_tune


class CaterObjectDetector(object):

    @staticmethod
    def remove_low_probability_object(model_output: dict, accuracy_threshold: float = 0.8) -> dict:
        scores = model_output["scores"]
        scores_mask = torch.sum((scores >= accuracy_threshold)).item()

        boxes = model_output["boxes"][:scores_mask, :]
        labels = model_output["labels"][:scores_mask]
        scores = model_output["scores"][:scores_mask]

        high_prob_output = {
            "boxes": boxes,
            "labels": labels,
            "scores": scores
        }

        return high_prob_output

    @staticmethod
    def get_label_bb(output_prediction: Dict[str, np.ndarray], label: int) -> Tuple[Tuple[int, int, int, int], Tuple[int, int , int, int]]:
        labels = output_prediction["labels"]
        label_indices = np.where(labels == label)[0]

        if len(label_indices) == 0:  # label was not found in frame
            return (-1, -1, -1, -1), (-1, -1, -1, -1)

        else:
            label_index = label_indices[0]  # take only first appearance if there are more than 1
            label_boxes = output_prediction["bb"][label_index]

            cx = (label_boxes[0] + label_boxes[2]) // 2
            cy = (label_boxes[1] + label_boxes[3]) // 2
            w_box = label_boxes[2] - label_boxes[0]
            h_box = label_boxes[3] - label_boxes[1]

        return (cx, cy, w_box, h_box), label_boxes

    def __init__(self, saved_detector_path, class_names_to_indices: dict):
        self.saved_detector_path = saved_detector_path
        self.num_classes = 193
        self.indices_to_names = {index: name for name, index in class_names_to_indices.items()}
        self.detector: nn.Module = None

    def load_model(self, compute_device: torch.device) -> None:

        # define base model
        detector: nn.Module = get_fast_rcnn_for_fine_tune(self.num_classes)

        # load saved parameters
        saved_parameters = torch.load(self.saved_detector_path)
        saved_weights = saved_parameters["model_state_dict"]
        detector.load_state_dict(saved_weights)

        # move to predict model and load to compute device
        detector.eval()
        detector.to(compute_device)

        self.detector = detector

    def __call__(self, frame: np.ndarray, compute_device: torch.device) -> dict:
        with torch.no_grad():

            # convert frame to RGB and normalize
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame / 256

            # transform to tensor and add batch dimension
            frame_tensor = torch.as_tensor(frame, dtype=torch.float32)
            frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)

            # load to device and predict
            frame_tensor = frame_tensor.to(compute_device)
            y = self.detector(frame_tensor)

        return y

    def save_detector_output(self, save_path: str, input_image: np.ndarray, output_prediction: dict) -> None:
        # first convert input image to rgb

        # add bounding boxes and descriptions
        labels_and_boxes = zip(output_prediction["labels"], output_prediction["boxes"])
        for label, bb in labels_and_boxes:

            # make sure label and bb reside on cpu
            label = label.item()
            bb = bb.cpu().numpy().astype('int32')

            # transform label from index to name
            label = self.indices_to_names[label]

            # add bounding box and its description
            input_image = cv2.rectangle(input_image, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 0), 1)
            cv2.putText(input_image, label, (bb[0], bb[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (36, 255, 12), 1)

        # save to file or show
        cv2.imwrite(save_path, input_image)

    def get_last_frame_detector_predict_object(self, object_id: int, video_path: str, compute_device: torch.device, predict_batch_size: int = 24) -> Tuple[int, int]:
        last_frame_with_object = 0
        current_frame_id = 0
        video_frames = []

        # capture video and verify it worked as expected
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise 'Unable to open video {}'.format(video_path)

        flag, frame = cap.read()
        while flag:
            current_frame_id += 1

            # transform to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frames.append(frame)
            flag, frame = cap.read()

        video_tensor = torch.as_tensor(video_frames, dtype=torch.float32)
        video_tensor = video_tensor.permute(0, 3, 1, 2)  # transform to dimension (num_frames, channels, width, height)
        video_tensor = video_tensor / 256

        # create batches for model using torch.chunk
        num_frames = len(video_frames)
        num_chunks = len(video_frames) // predict_batch_size
        if num_frames % predict_batch_size != 0:
            num_chunks += 1  # last batch is smaller
        frames_batches = torch.chunk(video_tensor, chunks=num_chunks, dim=0)

        with torch.no_grad():
            batch_start_index = 0
            for batch in frames_batches:
                batch = batch.to(compute_device)
                y = self.detector(batch)

                # check if object appears with high probability
                high_prob_y = map(lambda x: self.remove_low_probability_object(x), y)
                object_mask = np.array(list(map(lambda x: object_id in x["labels"], high_prob_y)))
                if sum(object_mask) != 0:
                    # last_appearance_in_batch = len(snitch_mask) - snitch_mask[::-1].index(True) - 1
                    last_appearance_in_batch = np.where(object_mask)[0][-1]  # last appearance, there's got to be at least one
                    last_frame_with_object = batch_start_index + last_appearance_in_batch
                batch_start_index += len(batch)

        # add 1 to last frame index to keep consistency with tracker - starting frames count from 1
        last_frame_with_object += 1

        # return also number of total frames in video
        return last_frame_with_object, num_frames

