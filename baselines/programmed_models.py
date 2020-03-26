from typing import Dict, List, Any

import torch
import numpy as np

from baselines.detector import CaterObjectDetector
from baselines.DaSiamRPN.code.run_SiamRPN import SiamRPN_init, SiamRPN_track


class AbstractReasoner(object):

    def __init__(self, index_to_track: int):
        self.index_to_track = index_to_track
        self.state: dict = {
            "target_pos": (-1, 1),
            "target_sz": (0, 0),
            "snitch_box": [-1, -1, -1, -1]
        }
        self.snitch_visible = False

    def track_for_frame(self, frame: np.ndarray, frame_index: int, frames_predictions: Dict[str, List[np.ndarray]], video_name:str = None) -> None:
        raise NotImplementedError("A tracker must implement track for frame method")


class ObjectDetectWithSiamTracker(AbstractReasoner):

    def __init__(self, index_to_track: int, tracker_net: torch.nn.Module, compute_device: torch.device):
        super().__init__(index_to_track)
        self.tracker = tracker_net
        self.tracker_initiated = False
        self.compute_device = compute_device
        self.siam_tracker_state = None

    def track_for_frame(self, frame: np.ndarray, frame_index: int, frames_predictions: Dict[str, List[np.ndarray]], video_name:str = None) -> None:

        current_frame_bb = frames_predictions["bb"][frame_index]
        current_frame_label = frames_predictions["labels"][frame_index]
        frame_prediction = {
            "bb": current_frame_bb,
            "labels": current_frame_label
        }
        (cx, cy, w_box, h_box), (x1, y1, x2, y2) = CaterObjectDetector.get_label_bb(frame_prediction, self.index_to_track)

        # if object is detected just update the state to its location
        if cx >= 0 and cy >= 0:
            self.state["target_pos"] = (cx, cy)
            self.state["target_sz"] = (w_box, h_box)
            self.state["snitch_box"] = [x1, y1, x2, y2]
            self.tracker_initiated = False
            self.snitch_visible = True

        else:
            # object is not detected - apply reasoning using tracker network
            self.snitch_visible = False

            # if tracker network is not initiated - initiate it
            # this will only happen in the first frame because the snitch will always appear on the first frame
            if not self.tracker_initiated:
                target_pos = np.array(self.state["target_pos"])
                target_sz = np.array(self.state["target_sz"])

                self.siam_tracker_state = SiamRPN_init(frame, target_pos, target_sz, self.tracker, self.compute_device)
                self.tracker_initiated = True

            # anyway track it using siam tracker
            self.siam_tracker_state = SiamRPN_track(self.siam_tracker_state, frame, self.compute_device)  # track
            self.state["target_pos"] = self.siam_tracker_state["target_pos"]
            self.state["target_sz"] = self.siam_tracker_state["target_sz"]


class HeuristicReasoner(AbstractReasoner):

    def __init__(self, index_to_track: int):
        super().__init__(index_to_track)
        self.stack = []

    def track_for_frame(self, frame: np.ndarray, frame_index: int, frames_predictions: Dict[str, List[np.ndarray]], video_name:str = None) -> None:
        try:
            current_frame_bb = frames_predictions["bb"][frame_index]
            current_frame_label = frames_predictions["labels"][frame_index]
            frame_prediction = {
                "bb": current_frame_bb,
                "labels": current_frame_label
            }

            # first we are looking for the snitch
            (cx, cy, w_box, h_box), (x1, y1, x2, y2) = CaterObjectDetector.get_label_bb(frame_prediction, self.index_to_track)

            # if snitch is found - update state and init stack (forget about previous state)
            if cx >= 0 and cy >= 0:
                self.snitch_visible = True
                self.state["snitch_box"] = [x1, y1, x2, y2]
                self._update_state(cx, cy, w_box, h_box, self.index_to_track)
                self.stack = []  # init stack

            # we are tacking the snitch
            elif len(self.stack) == 0:
                self.snitch_visible = False

                closest_label_index = self._get_index_of_closest_object(frame_prediction)
                (cx, cy, w_box, h_box), _ = CaterObjectDetector.get_label_bb(frame_prediction, closest_label_index)
                self._update_state(cx, cy, w_box, h_box, closest_label_index)
                self.stack.append(self.index_to_track)

            # we are tracking another object, not the snitch
            else:
                self.snitch_visible = False

                current_index_to_track = self.state["object_label"]
                (cx, cy, w_box, h_box), _ = CaterObjectDetector.get_label_bb(frame_prediction, current_index_to_track)

                # if we didn't find the object we are looking for - it means some other object contains it
                if cx < 0 and cy < 0:

                    closest_label_index = self._get_index_of_closest_object(frame_prediction)
                    (cx, cy, w_box, h_box), _ = CaterObjectDetector.get_label_bb(frame_prediction, closest_label_index)

                    # for sure this object is present, no need to check because this is how we found it
                    self._update_state(cx, cy, w_box, h_box, closest_label_index)
                    self.stack.append(current_index_to_track)

                # if we found it - we have to check if it is still covering the object that is before him on the stack
                # option 1 - if the object before it is not found - he is still covering it
                # option 2- if the object before it is found - just keep tracking it
                else:
                    previous_index_to_track = self.stack[-1]
                    (prev_cx, prev_cy, prev_w_box, prev_h_box), _ = CaterObjectDetector.get_label_bb(frame_prediction, previous_index_to_track)

                    # we found a previous object - we are in option 2
                    if prev_cx >= 0 and prev_cy >= 0:
                        self._update_state(prev_cx, prev_cy, prev_w_box, prev_h_box, previous_index_to_track)
                        self.stack.pop(-1)

                    # didn't find previous object - we are in option 1
                    else:
                        self._update_state(cx, cy, w_box, h_box, current_index_to_track)
        except ValueError:
            print(f"value error in frame {frame_index}, skipping action for this frame (snitch position is not updated")

    def _get_index_of_closest_object(self, output_predictions: Dict[str, np.ndarray]) -> int:
        cx, cy = self.state["target_pos"]
        current_object_center = np.array([cx, cy])

        all_boxes = output_predictions["bb"]
        all_centers = list(map(lambda x: self._get_cx_cy_from_bb_output(x), all_boxes))
        all_distances = np.linalg.norm(all_centers - current_object_center, axis=1)

        closest_index = np.argmin(all_distances).item()
        closest_label = output_predictions["labels"][closest_index].item()

        return closest_label

    def _get_cx_cy_from_bb_output(self, boxes: np.ndarray) -> np.ndarray:
        cx = (boxes[0] + boxes[2]) // 2
        cy = (boxes[1] + boxes[3]) // 2

        center = np.array([cx, cy])
        return center

    def _update_state(self, cx: int, cy: int, w_box: int, h_box: int, object_label: int) -> None:
        self.state["target_pos"] = (cx, cy)
        self.state["target_sz"] = (w_box, h_box)
        self.state["object_label"] = object_label

        # object_sz holds the size the global object we are tracking - aka the snitch
        if object_label == self.index_to_track:
            self.state["object_sz"] = (w_box, h_box)
