import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
from functools import cmp_to_key

import numpy as np
import torch
from torch.utils.data import Dataset

from object_indices import is_cone_object

SNITCH_NAME = "small_gold_spl_metal_Spl_0"
SNITCH_INDEX = 140
VIDEO_NUM_FRAMES = 300
SNITCH_INPUT_TRACKER_INDEX = 0


class CaterAbstractDataset(Dataset):
    def __init__(self, predictions_dir: str, label_dir: str):
        self.predictions_dir: Path = Path(predictions_dir)
        self.labels_dir: Path = Path(label_dir)

        # add video names and labels data
        self.videos_names: List[str] = []
        self.label_paths: Dict[str, str] = {}

        # extra variables
        self.max_objects: int = 15
        self.frame_shapes = np.array([320, 240, 320, 240])  # width, height, width, height
        self.frame_shapes_for_labels = np.array([320, 240, 320, 240])

    def _load_snitch_labels_for_video(self, video_name: str) -> np.ndarray:
        video_labels_path = self.label_paths[video_name]
        with open(video_labels_path, "rb") as f:
            video_labels: Dict[str, List[List[int]]] = json.load(f)

        snitch_labels = video_labels[SNITCH_NAME]

        # convert from x,y,w,h to x,y,x,y
        snitch_labels = [[x, y, x + w, y + h] for x, y, w, h in snitch_labels]

        # normalize label to be between 0 and 1 (frame ratio)
        snitch_labels = np.array(snitch_labels) / self.frame_shapes_for_labels
        return np.array(snitch_labels)

    def object_indices_comparator(self, idx1, idx2):
        # snitch will always appear first
        if idx1 == SNITCH_INDEX:
            return -1
        elif idx2 == SNITCH_INDEX:
            return 1
        else:
            return idx1 - idx2

    def object_and_bb_comparator(self, obj1, obj2):
        idx1, idx2 = obj1[0], obj2[0]
        return self.object_indices_comparator(idx1, idx2)

    def _load_predictions_pkl(self, prediction_path: str) -> Dict[str, List[np.ndarray]]:
        with open(prediction_path, 'rb') as f:
            prediction_data = pickle.load(f)

        return prediction_data

    def _init_dataset_if_not_initiated(self) -> None:
        # if not initiated
        if len(self.videos_names) == 0:

            # init video names
            predictions_files = list(Path(self.predictions_dir).glob("*.pkl"))
            video_names = [str(file_.stem) for file_ in predictions_files]

            video_names = sorted(video_names)
            self.videos_names = video_names

            # init video label paths
            for video_name in self.videos_names:
                video_labels_path = self.labels_dir / (video_name + "_bb.json")
                self.label_paths[video_name] = str(video_labels_path)

    def _get_all_video_objects(self, object_labels: List[np.ndarray]) -> List[int]:
        video_objects = set()
        for frame_idx in range(len(object_labels)):
            frame_objects = object_labels[frame_idx]
            video_objects = video_objects.union(frame_objects)

        return list(video_objects)

    def _get_objects_center_distances_to_tracks(self, frame_boxes: np.ndarray, compared_center_location: np.ndarray) -> np.ndarray:
        frame_boxes_centers = list(map(self._get_bounding_box_center, frame_boxes))
        all_distances = np.linalg.norm(frame_boxes_centers - compared_center_location, axis=1)

        new_frame_boxes = np.zeros((frame_boxes.shape[0], frame_boxes.shape[1] + 1))
        new_frame_boxes[:, :-1] = frame_boxes
        new_frame_boxes[:, -1] = all_distances

        return new_frame_boxes

    def _get_index_of_closes_object(self, frame_boxes: np.ndarray, last_location: np.ndarray) -> int:

        frame_boxes_centers = list(map(self._get_bounding_box_center, frame_boxes))
        last_location_center = self._get_bounding_box_center(last_location)

        all_distances = np.linalg.norm(frame_boxes_centers - last_location_center, axis=1)
        closest_index = np.argmin(all_distances).item()

        return closest_index

    def _get_bounding_box_center(self, box: np.ndarray) -> np.ndarray:
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        center = np.array([cx, cy])

        return center

    def __len__(self) -> int:
        self._init_dataset_if_not_initiated()
        return len(self.videos_names)

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.tensor, torch.tensor], Tuple[torch.tensor, torch.tensor], str]:
        raise NotImplementedError("get item method must be implemented for dataset")


class CaterAbstract5TracksForObjectsDataset(CaterAbstractDataset):
    def __init__(self, predictions_dir: str, label_dir: str):
        super().__init__(predictions_dir, label_dir)
        self.frame_shapes = np.array([320, 240, 320, 240, 1])  # width, height, width, height and visibility flag

    def _normalize_and_pad_predictions(self, prediction_boxes: List[np.ndarray], object_labels: List[np.ndarray]) -> List[np.ndarray]:
        padded_video_boxes: List[np.ndarray] = []
        padding_bb = np.array([0] * 5)
        frames_dimensions = self.frame_shapes

        num_frames = len(object_labels)
        video_objects = self._get_all_video_objects(object_labels)
        sorted_objects = sorted(video_objects, key=cmp_to_key(self.object_indices_comparator))
        video_objects_order = {idx: label for idx, label in enumerate(sorted_objects)}
        num_possible_objects_video = min(len(video_objects_order), self.max_objects)

        for frame_idx in range(num_frames):

            padded_frame_boxes: List[np.ndarray] = []
            frame_objects: np.ndarray = object_labels[frame_idx]
            frame_predictions: np.ndarray = prediction_boxes[frame_idx]
            objects_and_bb: List[Tuple[int, np.ndarray]] = list(zip(frame_objects, frame_predictions))
            sorted_objects_and_bb = sorted(objects_and_bb, key=cmp_to_key(self.object_and_bb_comparator))
            num_objects_in_frame = len(sorted_objects_and_bb)

            current_object_idx = 0
            video_objects_order_idx = 0
            last_object = -1
            while current_object_idx < num_objects_in_frame:
                # if the video frame contains more than max objects we will discard the last ones
                # it is considered a limitation of the perception model
                if video_objects_order_idx >= num_possible_objects_video:
                    break

                current_object, current_bb = sorted_objects_and_bb[current_object_idx]

                if current_object == video_objects_order[video_objects_order_idx]:
                    # the objects are in the correct order (no missing objects)

                    # add real bb bit set to 1
                    object_tracks = np.append(current_bb, 1)
                    padded_frame_boxes.append(object_tracks)
                    current_object_idx += 1
                    video_objects_order_idx += 1
                    last_object = current_object

                elif current_object == last_object:
                    # in case we have the same object id more than once this is a mistake of the perception model
                    # we will ignore that object
                    current_object_idx += 1

                else:
                    # object is missing, needs to add padding instead

                    padded_frame_boxes.append(padding_bb)  # including padding bb bit set to 0
                    video_objects_order_idx += 1

            # if necessary add more padding bb
            num_current_frame_objects = len(padded_frame_boxes)
            if num_current_frame_objects < self.max_objects:
                required_padding_length = self.max_objects - num_current_frame_objects
                padding_bbs = [padding_bb] * required_padding_length
                padded_frame_boxes.extend(padding_bbs)

            # finally this is the bounding boxes samples for the current frame
            # pack as a numpy array with dimensions (10, 5)
            # normalize and then add to list videos frame bbs
            norm_frame_boxes = np.array(padded_frame_boxes)
            norm_frame_boxes = norm_frame_boxes / frames_dimensions  # normalize
            padded_video_boxes.append(norm_frame_boxes)

        return padded_video_boxes


    def _get_closest_object_to_track_vector(self, padded_video_boxes: List[np.ndarray]) -> List[int]:
        object_index_to_track: List[int] = []
        containment_stack: List[int] = []
        last_location: np.ndarray = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        current_object_to_track = SNITCH_INPUT_TRACKER_INDEX

        for frame_boxes in padded_video_boxes:
            snitch_visible_bit = frame_boxes[SNITCH_INPUT_TRACKER_INDEX, 4]
            if snitch_visible_bit:
                object_index_to_track.append(SNITCH_INPUT_TRACKER_INDEX)
                last_location = frame_boxes[SNITCH_INPUT_TRACKER_INDEX]
                current_object_to_track = SNITCH_INPUT_TRACKER_INDEX
                containment_stack = []

            # we were tracking the snitch up until now
            # but now it is not visible anymore
            elif current_object_to_track == SNITCH_INPUT_TRACKER_INDEX:
                # get the index of the object that is the closest object to the snitch last known location
                closest_index = self._get_index_of_closes_object(frame_boxes, last_location)

                object_index_to_track.append(closest_index)
                last_location = frame_boxes[closest_index]
                current_object_to_track = closest_index
                containment_stack.append(SNITCH_INPUT_TRACKER_INDEX)

            # we are tracking another object, not the snitch
            # we need to see if it is still visible or not
            else:
                tracked_object_visible_bit = frame_boxes[current_object_to_track, 4]

                # if we didn't find this object we will apply the same logic recursively
                if not tracked_object_visible_bit:
                    closest_index = self._get_index_of_closes_object(frame_boxes, last_location)

                    object_index_to_track.append(closest_index)
                    last_location = frame_boxes[closest_index]
                    containment_stack.append(current_object_to_track)
                    current_object_to_track = closest_index

                # the object we are tracking is still visible
                # but we have to check if the object it contained is visible or not
                else:
                    previous_index_to_track = containment_stack[-1]
                    previous_object_visible_bit = frame_boxes[previous_index_to_track, 4]

                    # we now see a object the was contained or occluded
                    if previous_object_visible_bit:
                        object_index_to_track.append(previous_index_to_track)
                        last_location = frame_boxes[previous_index_to_track]
                        current_object_to_track = previous_index_to_track
                        containment_stack.pop(-1)

                    # we still do not see an object that was contained or occluded
                    # we will keep tracker the object we are tracking now
                    else:
                        object_index_to_track.append(current_object_to_track)
                        last_location = frame_boxes[current_object_to_track]

        return object_index_to_track


class CaterAbstract6TracksForObjectsDataset(CaterAbstractDataset):
    def __init__(self, predictions_dir: str, label_dir: str):
        super().__init__(predictions_dir, label_dir)
        self.frame_shapes = np.array([320, 240, 320, 240, 1, 1])  # width, height, width, height and visibility flag, is cone flag

    def _normalize_and_pad_predictions(self, prediction_boxes: List[np.ndarray], object_labels: List[np.ndarray]) -> List[np.ndarray]:
        padded_video_boxes: List[np.ndarray] = []
        padding_bb = np.array([0] * 6)  # 6 tracks for object, not cone
        padding_cone_bb = np.array([0, 0, 0, 0, 0, 1])  # 6 tracks for cone objects

        num_frames = len(object_labels)
        video_objects = self._get_all_video_objects(object_labels)
        sorted_objects = sorted(video_objects, key=cmp_to_key(self.object_indices_comparator))
        video_objects_order = {idx: label for idx, label in enumerate(sorted_objects)}
        num_possible_objects_video = min(len(video_objects_order), self.max_objects)

        for frame_idx in range(num_frames):

            padded_frame_boxes: List[np.ndarray] = []
            frame_objects: np.ndarray = object_labels[frame_idx]
            frame_predictions: np.ndarray = prediction_boxes[frame_idx]
            objects_and_bb: List[Tuple[int, np.ndarray]] = list(zip(frame_objects, frame_predictions))
            sorted_objects_and_bb = sorted(objects_and_bb, key=cmp_to_key(self.object_and_bb_comparator))
            num_objects_in_frame = len(sorted_objects_and_bb)

            current_object_idx = 0
            video_objects_order_idx = 0
            last_object = -1
            while current_object_idx < num_objects_in_frame:
                # if the video frame contains more than max objects we will discard the last ones
                # it is considered a limitation of the perception model
                if video_objects_order_idx >= num_possible_objects_video:
                    break

                current_object, current_bb = sorted_objects_and_bb[current_object_idx]

                if current_object == video_objects_order[video_objects_order_idx]:
                    # the objects are in the correct order (no missing objects)

                    # add real bb bit set to 1, and also is cone bit
                    object_tracks = np.append(current_bb, [1, is_cone_object(current_object)])
                    padded_frame_boxes.append(object_tracks)
                    current_object_idx += 1
                    video_objects_order_idx += 1
                    last_object = current_object

                elif current_object == last_object:
                    # in case we have the same object id more than once this is a mistake of the perception model
                    # we will ignore that object
                    current_object_idx += 1

                else:
                    # object is missing, needs to add padding instead

                    # check if the object that is supposed to be in this index is a cone or not
                    if is_cone_object(video_objects_order[video_objects_order_idx]):
                        padded_frame_boxes.append(padding_cone_bb)
                    else:
                        padded_frame_boxes.append(padding_bb)  # including padding bb bit set to 0

                    video_objects_order_idx += 1

            # if necessary add more padding bb
            num_current_frame_objects = len(padded_frame_boxes)
            if num_current_frame_objects < self.max_objects:
                required_padding_length = self.max_objects - num_current_frame_objects
                padding_bbs = [padding_bb] * required_padding_length
                padded_frame_boxes.extend(padding_bbs)

            # finally this is the bounding boxes samples for the current frame
            # pack as a numpy array with dimensions (10, 5)
            # normalize and then add to list videos frame bbs
            norm_frame_boxes = np.array(padded_frame_boxes)
            norm_frame_boxes = norm_frame_boxes / self.frame_shapes  # normalize
            padded_video_boxes.append(norm_frame_boxes)

        return padded_video_boxes

    def _get_closest_object_to_track_vector(self, padded_video_boxes: List[np.ndarray]) -> List[int]:
        object_index_to_track: List[int] = []
        containment_stack: List[int] = []
        last_location: np.ndarray = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        current_object_to_track = SNITCH_INPUT_TRACKER_INDEX

        for frame_boxes in padded_video_boxes:
            snitch_visible_bit = frame_boxes[SNITCH_INPUT_TRACKER_INDEX, 4]
            if snitch_visible_bit:
                object_index_to_track.append(SNITCH_INPUT_TRACKER_INDEX)
                last_location = frame_boxes[SNITCH_INPUT_TRACKER_INDEX]
                current_object_to_track = SNITCH_INPUT_TRACKER_INDEX
                containment_stack = []

            # we were tracking the snitch up until now
            # but now it is not visible anymore
            elif current_object_to_track == SNITCH_INPUT_TRACKER_INDEX:
                # get the index of the object that is the closest object to the snitch last known location
                closest_index = self._get_index_of_closes_object(frame_boxes, last_location)

                # now check if it is a cone object or not
                # if yes - probably containment, if not - probably occlusion and we want to keep tracking the snitch
                is_cone_bit = frame_boxes[closest_index, 5]

                if is_cone_bit:
                    # it is a cone object - so probably containment
                    object_index_to_track.append(closest_index)
                    last_location = frame_boxes[closest_index]
                    current_object_to_track = closest_index
                    containment_stack.append(SNITCH_INPUT_TRACKER_INDEX)
                else:
                    # it is probably occlusion - we want to keep track the snitch
                    # so we will keep our last location as is, without update (the last place we saw the snitch)
                    object_index_to_track.append(SNITCH_INPUT_TRACKER_INDEX)
                    current_object_to_track = SNITCH_INPUT_TRACKER_INDEX

            # we are tracking another object, not the snitch
            # we need to see if it is still visible or not
            else:
                tracked_object_visible_bit = frame_boxes[current_object_to_track, 4]

                # if we didn't find this object we will apply the same logic recursively
                if not tracked_object_visible_bit:
                    closest_index = self._get_index_of_closes_object(frame_boxes, last_location)
                    is_cone_bit = frame_boxes[closest_index, 5]

                    if is_cone_bit:

                        object_index_to_track.append(closest_index)
                        last_location = frame_boxes[closest_index]
                        containment_stack.append(current_object_to_track)
                        current_object_to_track = closest_index

                    else:

                        object_index_to_track.append(current_object_to_track)
                        # no change in current object to track
                        # no change in last location

                # the object we are tracking is still visible
                # but we have to check if the object it contained is visible or not
                else:
                    previous_index_to_track = containment_stack[-1]
                    previous_object_visible_bit = frame_boxes[previous_index_to_track, 4]

                    # we now see a object the was contained or occluded
                    if previous_object_visible_bit:
                        containment_stack.pop(-1)
                        object_index_to_track.append(previous_index_to_track)
                        last_location = frame_boxes[previous_index_to_track]
                        current_object_to_track = previous_index_to_track

                    # we still do not see an object that was contained or occluded
                    # we will keep tracker the object we are tracking now
                    else:
                        object_index_to_track.append(current_object_to_track)
                        last_location = frame_boxes[current_object_to_track]

        return object_index_to_track


class Cater5TracksForObjectsInferenceDataset(CaterAbstract5TracksForObjectsDataset):
    def __init__(self, predictions_dir: str, label_dir: str):
        super().__init__(predictions_dir, label_dir)

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.tensor, torch.tensor], Tuple[torch.tensor, torch.tensor], str]:
        self._init_dataset_if_not_initiated()

        video_name = self.videos_names[idx]

        # load labels
        snitch_labels: np.ndarray = self._load_snitch_labels_for_video(video_name)

        # load predictions
        video_predictions_path = str(self.predictions_dir / (video_name + ".pkl"))
        prediction_data = self._load_predictions_pkl(video_predictions_path)

        prediction_boxes: List[np.ndarray] = prediction_data["bb"]
        objects: List[np.ndarray] = prediction_data["labels"]

        # normalize relative to frames dimensions
        # add padding where number of objects is smaller then the maximum
        prediction_boxes = self._normalize_and_pad_predictions(prediction_boxes, objects)
        object_to_track = self._get_closest_object_to_track_vector(prediction_boxes)

        # transform to tensors and return
        tensor_boxes: torch.tensor = torch.tensor(prediction_boxes, dtype=torch.float32)
        tensor_index_to_track: torch.tensor = torch.tensor(object_to_track, dtype=torch.int64)
        tensor_labels: torch.tensor = torch.tensor(snitch_labels, dtype=torch.float32)
        empty_tensor: torch.tensor = torch.tensor([])

        return (tensor_boxes, tensor_index_to_track), (tensor_labels, empty_tensor), video_name


class Cater5TracksForObjectsTrainingDataset(CaterAbstract5TracksForObjectsDataset):
    def __init__(self, predictions_dir: str, label_dir: str, mask_annotations_path: str):
        super().__init__(predictions_dir, label_dir)
        self.mask_annotations_path: str = mask_annotations_path

        # add labels and mask attributes
        self.mask_frames: Dict[str, np.ndarray] = {}

    def _init_dataset_if_not_initiated(self) -> None:
        # if not initiated
        if len(self.videos_names) == 0:
            super()._init_dataset_if_not_initiated()

            # init occlusions annotations
            with open(self.mask_annotations_path, "r") as f:
                for line in f:
                    line = line[:-1]
                    video_name, mask_frames_str = line.split(sep="\t")
                    if video_name in self.videos_names:
                        if len(mask_frames_str) == 0:  # no matching frames in video
                            self.mask_frames[video_name] = np.array([], dtype=np.bool)
                        else:
                            mask_frames: List[str] = mask_frames_str.split(sep=",")
                            self.mask_frames[video_name] = np.array(mask_frames, dtype=np.int)

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.tensor, torch.tensor], Tuple[torch.tensor, torch.tensor], str]:
        self._init_dataset_if_not_initiated()

        video_name = self.videos_names[idx]

        # load labels
        snitch_labels: np.ndarray = self._load_snitch_labels_for_video(video_name)

        # get occlusions annotations
        mask_frames: np.ndarray = self.mask_frames[video_name]
        mask: np.ndarray = np.zeros((VIDEO_NUM_FRAMES, 4), dtype=np.bool)
        mask[mask_frames, :] = True

        # load predictions
        video_predictions_path = str(self.predictions_dir / (video_name + ".pkl"))
        prediction_data = self._load_predictions_pkl(video_predictions_path)

        prediction_boxes: List[np.ndarray] = prediction_data["bb"]
        objects: List[np.ndarray] = prediction_data["labels"]

        # normalize relative to frames dimensions
        # add padding where number of objects is smaller then the maximum
        prediction_boxes = self._normalize_and_pad_predictions(prediction_boxes, objects)
        object_to_track = self._get_closest_object_to_track_vector(prediction_boxes)

        # transform to tensors and return
        tensor_boxes: torch.tensor = torch.tensor(prediction_boxes, dtype=torch.float32)
        tensor_index_to_track: torch.tensor = torch.tensor(object_to_track, dtype=torch.int64)
        tensor_frame_mask: torch.tensor = torch.tensor(mask)
        tensor_labels: torch.tensor = torch.tensor(snitch_labels, dtype=torch.float32)

        return (tensor_boxes, tensor_index_to_track), (tensor_labels, tensor_frame_mask), video_name


class Cater6TracksForObjectsTrainingDataset(CaterAbstract6TracksForObjectsDataset):
    def __init__(self, predictions_dir: str, label_dir: str, mask_annotations_path: str):
        super().__init__(predictions_dir, label_dir)
        self.mask_annotations_path: str = mask_annotations_path

        # add labels and mask attributes
        self.mask_frames: Dict[str, np.ndarray] = {}

    def _init_dataset_if_not_initiated(self) -> None:
        # if not initiated
        if len(self.videos_names) == 0:
            super()._init_dataset_if_not_initiated()

            # init occlusions annotations
            with open(self.mask_annotations_path, "r") as f:
                for line in f:
                    line = line[:-1]
                    video_name, mask_frames_str = line.split(sep="\t")
                    if video_name in self.videos_names:
                        if len(mask_frames_str) == 0:  # no matching frames in video
                            self.mask_frames[video_name] = np.array([], dtype=np.bool)
                        else:
                            mask_frames: List[str] = mask_frames_str.split(sep=",")
                            self.mask_frames[video_name] = np.array(mask_frames, dtype=np.int)

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.tensor, torch.tensor], Tuple[torch.tensor, torch.tensor], str]:
        self._init_dataset_if_not_initiated()

        video_name = self.videos_names[idx]

        # load labels
        snitch_labels: np.ndarray = self._load_snitch_labels_for_video(video_name)

        # get occlusions annotations
        mask_frames: np.ndarray = self.mask_frames[video_name]
        mask: np.ndarray = np.zeros((VIDEO_NUM_FRAMES, 4), dtype=np.bool)
        mask[mask_frames, :] = True

        # load predictions
        video_predictions_path = str(self.predictions_dir / (video_name + ".pkl"))
        prediction_data = self._load_predictions_pkl(video_predictions_path)

        prediction_boxes: List[np.ndarray] = prediction_data["bb"]
        objects: List[np.ndarray] = prediction_data["labels"]

        # normalize relative to frames dimensions
        # add padding where number of objects is smaller then the maximum
        prediction_boxes = self._normalize_and_pad_predictions(prediction_boxes, objects)
        object_to_track = self._get_closest_object_to_track_vector(prediction_boxes)

        # transform to tensors and return
        tensor_boxes: torch.tensor = torch.tensor(prediction_boxes, dtype=torch.float32)
        tensor_index_to_track: torch.tensor = torch.tensor(object_to_track, dtype=torch.int64)
        tensor_frame_mask: torch.tensor = torch.tensor(mask)
        tensor_labels: torch.tensor = torch.tensor(snitch_labels, dtype=torch.float32)

        return (tensor_boxes, tensor_index_to_track), (tensor_labels, tensor_frame_mask), video_name


class Cater6TracksForObjectsInferenceDataset(CaterAbstract6TracksForObjectsDataset):
    def __init__(self, predictions_dir: str, label_dir: str):
        super().__init__(predictions_dir, label_dir)

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.tensor, torch.tensor], Tuple[torch.tensor, torch.tensor], str]:
        self._init_dataset_if_not_initiated()

        video_name = self.videos_names[idx]

        # load labels
        snitch_labels: np.ndarray = self._load_snitch_labels_for_video(video_name)

        # load predictions
        video_predictions_path = str(self.predictions_dir / (video_name + ".pkl"))
        prediction_data = self._load_predictions_pkl(video_predictions_path)

        prediction_boxes: List[np.ndarray] = prediction_data["bb"]
        objects: List[np.ndarray] = prediction_data["labels"]

        # normalize relative to frames dimensions
        # add padding where number of objects is smaller then the maximum
        prediction_boxes = self._normalize_and_pad_predictions(prediction_boxes, objects)
        object_to_track = self._get_closest_object_to_track_vector(prediction_boxes)

        # transform to tensors and return
        tensor_boxes: torch.tensor = torch.tensor(prediction_boxes, dtype=torch.float32)
        tensor_index_to_track: torch.tensor = torch.tensor(object_to_track, dtype=torch.int64)
        tensor_labels: torch.tensor = torch.tensor(snitch_labels, dtype=torch.float32)
        empty_tensor: torch.tensor = torch.tensor([])

        return (tensor_boxes, tensor_index_to_track), (tensor_labels, empty_tensor), video_name
