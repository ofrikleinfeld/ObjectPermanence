from pathlib import Path
from typing import Dict, List, Any
import json
import pickle

import numpy as np
from tqdm import tqdm

from object_indices import OBJECTS_NAME_TO_IDX

NUM_FRAMES: int = 300
SNITCH_NAME: str = "Spl_0"
SNITCH_INDEX: int = 140


class VisibleObjectPredictions(object):

    @staticmethod
    def _load_json_file(json_path: str) -> Dict[str, Any]:
        with open(json_path, 'r') as f:
            data = json.load(f)

        return data

    @staticmethod
    def _cvt_obj_name_to_class(obj_name: str, video_data: dict) -> str:
        for obj in video_data["objects"]:
            if obj_name == obj["instance"]:
                return "_".join(obj[att] for att in ["size", "color", "shape", "material"])

    @staticmethod
    def _cvt_class_to_idx(class_names_dict, obj_name):
        split_name = obj_name.split("_Smooth")[0].split("_")
        try:
            obj_name = "_".join(split_name)
            return class_names_dict[f"{obj_name}"]
        except:
            obj_name = "_".join(split_name[:-2])
            return class_names_dict[f"{obj_name}"]

    @staticmethod
    def _prepare_bb_data(bb_data, object_labels, scene_data, object_names):
        combined_scene_data = {"bb": [], "labels": [], "3d_coord": [] }
        for i in range(NUM_FRAMES):
            frame_list = []
            coord_frame_list = []
            for obj_name, location_3d in zip(object_names, scene_data["objects"]):
                frame_list.append(bb_data[obj_name][i])
                coord_frame_list.append(location_3d["locations"][str(i)])
            combined_scene_data["bb"].append(frame_list)
            combined_scene_data["labels"].append(object_labels)
            combined_scene_data["3d_coord"].append(coord_frame_list)
        return combined_scene_data

    @staticmethod
    def _cvt_xywh_to_xyxy(bbox):
        return np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])

    @staticmethod
    def _cal_distance_from_cam(coords: List[List[float]]) -> List[float]:
        cam_loc: np.ndarray = np.array([7.1146, -6.1746, 5.5963])
        dist: List[float] = list(map(lambda x: np.linalg.norm(cam_loc - np.array(x)), coords))
        return dist

    @staticmethod
    def _get_contain_move_mask(video_output_path: str):

        with open(video_output_path, "r") as file:
            video_output_data: Dict[str, Any] = json.load(file)

        contain_and_move_list: List[bool] = [False] * NUM_FRAMES

        for obj_name, action_list in video_output_data['movements'].items():

            slide_flag: List[bool] = []
            pick_place_flag: List[bool] = []
            contain_flag: List[bool] = []

            if "Cone" in obj_name:

                for action in action_list:
                    contain_flag.append(("_contain" in action[0]) and (action[1] == SNITCH_NAME))
                    pick_place_flag.append("_pick_place" in action[0])
                    slide_flag.append("_slide" in action[0])

                if sum(contain_flag) != 0:
                    contain_index: np.ndarray = np.argwhere(contain_flag).flatten()
                    pick_place_index: np.ndarray = np.argwhere(pick_place_flag[contain_index[0]:]).flatten() + contain_index[0]
                    slide_index: np.ndarray = np.argwhere(slide_flag[contain_index[0]:]).flatten() + contain_index[0]

                    if len(contain_index) > len(pick_place_index):
                        pick_place_index: np.ndarray = np.append(pick_place_index, 1000)

                    for c_index, p_index in zip(contain_index, pick_place_index):

                        for s_index in slide_index:

                            if s_index >= p_index:
                                break

                            elif (s_index > c_index) and (s_index < p_index):
                                mask_len: int = action_list[s_index][3] + 1 - action_list[s_index][2]
                                contain_and_move_list[action_list[s_index][2]:action_list[s_index][3] + 1]\
                                    = mask_len * [True]

        return np.where(contain_and_move_list)[0]

    def __init__(self, pred_samples_path: str, video_scene_path: str, video_bb_path: str,
                 output_path: str, visible_ratio, perception_mode: str = "visible_only"):

        # relevant paths
        self.pred_samples_path: Path = Path(pred_samples_path)
        self.video_scene_path: Path = Path(video_scene_path)
        self.video_bb_path: Path = Path(video_bb_path)
        self.output_path: Path = Path(output_path)

        self.perception_mode: str = perception_mode

        self.visible_ratio = visible_ratio

        if self.perception_mode not in ["visible_only", "uncontained"]:
            raise NotImplementedError("This prediction_mode is not supported")

        # init predictions/video_names/class_names Handlers
        self.predictions: List[Dict[str, List[np.ndarray]]] = []
        self.video_names: List = []
        self.class_names: Dict[str, int] = {}
        self.scenes_data: List[Dict[str, Any]] = []
        self.gt_bb_data:  List[Dict[str, Any]] = []

    def _init_files_if_not_initiated(self, with_scenes_data=False) -> None:
        if len(self.video_names) == 0:
            self._init_video_names()
            self._init_class_names()
            if with_scenes_data:
                self._init_scenes_data()
                self._init_gt_bb_data()

    def _init_video_names(self) -> None:
        with open(str(self.pred_samples_path), "r") as file:
            for line in file:
                self.video_names.append(Path(line).stem)
        self.video_names = sorted(self.video_names)

    def _init_scenes_data(self) -> None:
        if len(self.scenes_data) == 0:
            for video_name in self.video_names:
                scene_output_file = str(self.video_scene_path / video_name) + ".json"
                self.scenes_data.append(self._load_json_file(scene_output_file))

    def _init_gt_bb_data(self) -> None:
        if len(self.gt_bb_data) == 0:
            for video_name in self.video_names:
                bb_gt_file = str(self.video_bb_path / video_name) + "_bb.json"
                self.gt_bb_data.append(self._load_json_file(bb_gt_file))

    def _init_class_names(self):
        self.class_names = OBJECTS_NAME_TO_IDX

    def _get_uncontained_bb(self, contain_obj_dict, scene_data, bb_data):
        not_contained_scene_data = {"bb": [], "labels": [], "3d_coord": []}
        contained_obj = [obj for obj in contain_obj_dict.keys()]  # index of contained objects
        object_names = ["_".join(obj[att] for att in ["size", "color", "shape", "material", "instance"]) for obj in scene_data["objects"]]
        object_labels = list(map(lambda x: self._cvt_class_to_idx(self.class_names, x), object_names))
        # iterating over the bounding boxes and labels
        combined_scene_data = self._prepare_bb_data(bb_data, object_labels, scene_data, object_names)

        for frame_num, (bb, label, coords) in enumerate(zip(combined_scene_data["bb"], combined_scene_data["labels"], combined_scene_data["3d_coord"])):
            frame_bb = []
            frame_label = []
            frame_coords = []
            # iterating over lists of bounding boxes and labels
            for obj_idx, obj_bb, obj_coords in zip(label, bb, coords):

                # check if object is contained
                if obj_idx not in contained_obj:
                    frame_bb.append(obj_bb)
                    frame_label.append(obj_idx)
                    frame_coords.append(obj_coords)

                # check if the current frame is within the containment occurrence
                else:
                    frame_contain_list: List = []
                    for frame_range in contain_obj_dict[obj_idx]:
                        frame_contain_list.append((frame_num < frame_range[0]) or (frame_num > frame_range[1]))
                    if all(frame_contain_list):
                        frame_bb.append(obj_bb)
                        frame_label.append(obj_idx)
                        frame_coords.append(obj_coords)

            # add the modified predictions to new_prediction dictionary
            not_contained_scene_data["labels"].append(np.array(frame_label))
            not_contained_scene_data["bb"].append(frame_bb)
            not_contained_scene_data["3d_coord"].append(frame_coords)

        return not_contained_scene_data

    def _check_if_obj_occluded(self, box_1: np.ndarray, box_2: np.ndarray, coord_1: List[float], coord_2: List[float],
                               overlap_thresh: float) -> Any:

        box_1: np.ndarray = self._cvt_xywh_to_xyxy(box_1)
        box_2: np.ndarray = self._cvt_xywh_to_xyxy(box_2)

        # obtain the X,Y of the intersection rectangle
        intersection_x_1: int = max(box_1[0], box_2[0])
        intersection_y_1: int = max(box_1[1], box_2[1])
        intersection_x_2: int = min(box_1[2], box_2[2])
        intersection_y_2: int = min(box_1[3], box_2[3])

        # calc the area of intersection rectangle
        intersection_area: int = max(0, intersection_x_2 - intersection_x_1 + 1) * max(0,
                                                                                       intersection_y_2 - intersection_y_1 + 1)

        # compute the area of both the prediction and ground-truth rectangles
        box_1_area: int = (box_1[2] - box_1[0] + 1) * (box_1[3] - box_1[1] + 1)
        box_2_area: int = (box_2[2] - box_2[0] + 1) * (box_2[3] - box_2[1] + 1)

        occ_ratio: float = intersection_area / min(box_1_area, box_2_area)

        if occ_ratio >= overlap_thresh:
            distance_from_cam: List[float] = self._cal_distance_from_cam([coord_1, coord_2])
            if (box_1_area < box_2_area) and (distance_from_cam[0] > distance_from_cam[1]):
                return [True, False]
            elif (box_2_area < box_1_area) and (distance_from_cam[1] > distance_from_cam[0]):
                return [False, True]
            else:
                return None
        else:
            return None

    def _get_objects_contained_frames(self, video_output_path: str, class_object_names) -> Dict[int, List[int]]:
        with open(video_output_path, "r") as file:
            video_output_data = json.load(file)

        contain_dict: Dict = {}

        for obj_name, action_list in video_output_data['movements'].items():

            contain_flag = ["_contain" in action[0] for action in action_list]
            pick_place_flag = ["_pick_place" in action[0] for action in action_list]

            if ("Cone" in obj_name) and (sum(contain_flag) != 0):
                # get the contain / pick place indexes
                contain_index = np.argwhere(contain_flag).flatten()

                pick_place_index = []
                for c_idx in contain_index:
                    p_flag = np.argwhere(pick_place_flag[c_idx:]).flatten()
                    if len(p_flag) > 0:
                        pick_place_index.append(p_flag[0] + c_idx)
                    else:
                        # padding with special index if pick place didn't occur before the end of the video
                        pick_place_index.append(-1)

                for c_idx, p_idx in zip(contain_index, pick_place_index):
                    obj_class_name = self._cvt_obj_name_to_class(action_list[c_idx][1], video_output_data)
                    obj_class_idx = self._cvt_class_to_idx(class_object_names, obj_class_name)
                    if obj_class_idx not in contain_dict.keys():
                        contain_dict[obj_class_idx] = []
                    contain_dict[obj_class_idx].append([action_list[c_idx][3],
                                                        NUM_FRAMES if p_idx == -1 else action_list[p_idx][2]])
        return contain_dict

    def _exclude_occluded_objects(self, frame_bb, frame_label, frame_coord, mode="all"):
        new_frame_bb: List[List[int]] = []
        new_frame_label: List[int] = []

        if mode == "all":
            object_occlusions_flags: Dict[int, bool] = {i: False for i in range(len(frame_label))}

            # looping over all possible bb pairs
            for i in range(len(frame_bb)):

                for j in range(i, len(frame_bb)):

                    bb_1, coord_1 = frame_bb[i], frame_coord[i]
                    bb_2, coord_2 = frame_bb[j], frame_coord[j]

                    occluded_results = self._check_if_obj_occluded(bb_1, bb_2, coord_1, coord_2, overlap_thresh=1 - self.visible_ratio)
                    if occluded_results is not None:
                        if occluded_results[0]:
                            object_occlusions_flags[i] = True
                        else:
                            object_occlusions_flags[j] = True

            for index, occlusion_flag in object_occlusions_flags.items():
                if not occlusion_flag:
                    new_frame_bb.append(frame_bb[index])
                    new_frame_label.append(frame_label[index])

            new_frame_labels_array = np.array(new_frame_label)
            return new_frame_bb, new_frame_labels_array

        elif mode == "snitch":

            snitch_place_idx = frame_label.tolist().index(SNITCH_INDEX)
            bb_1, coord_1 = frame_bb.pop(snitch_place_idx), frame_coord.pop(snitch_place_idx)

            for bb_2, coord_2 in zip(frame_bb, frame_coord):
                occluded_box = self._check_if_obj_occluded(bb_1, bb_2, coord_1, coord_2,
                                                           overlap_thresh=1-self.visible_ratio)

                if occluded_box is not None and occluded_box[0]:
                    return False

            return True

    def _get_visible_bb(self, objs_data: Dict[str, Any]) -> Dict[str, Any]:
        visible_obj_data = {"bb": [], "labels": []}
        all_frames_bb: List[List[np.ndarray]] = objs_data["bb"]
        all_frames_labels: List[np.ndarray] = objs_data["labels"]
        all_frames_coord = objs_data["3d_coord"]

        for frame_bb, frame_label, frame_coord in zip(all_frames_bb, all_frames_labels, all_frames_coord):
            # looping over all possible bb pairs
            frame_bb, frame_label = self._exclude_occluded_objects(frame_bb, frame_label, frame_coord)

            # we have to convert it to xy, xy instead of xy, wh to align with the object detection outputs
            frame_bb = [[object_bb[0], object_bb[1], object_bb[0] + object_bb[2], object_bb[1] + object_bb[3]] for object_bb in frame_bb]

            # append current frame data
            visible_obj_data["bb"].append(frame_bb)
            visible_obj_data["labels"].append(frame_label)

        return visible_obj_data

    def _save_new_scene_data(self, predictions: Dict[str, Any], video_name: Path):
        with open(str(self.output_path / video_name) + ".pkl", "wb") as f:
            pickle.dump(predictions, f)

    def generate_visible_predictions(self):
        self._init_files_if_not_initiated(with_scenes_data=True)

        for vid_name, scene_data, gt_bb_data in tqdm(zip(self.video_names, self.scenes_data, self.gt_bb_data), total=len(self.video_names)):
            contain_dict: Dict[int, List[int]] = self._get_objects_contained_frames(str(self.video_scene_path / vid_name) + ".json",
                                                              self.class_names)
            uncontained_objs_data = self._get_uncontained_bb(contain_dict, scene_data, gt_bb_data)

            if self.perception_mode == "uncontained":
                self._save_new_scene_data(uncontained_objs_data, vid_name)

            elif self.perception_mode == "visible_only":
                visible_obj_data = self._get_visible_bb(uncontained_objs_data)
                self._save_new_scene_data(visible_obj_data, vid_name)

    def generate_contain_and_move_mask(self):
        with open(str(self.output_path / f"{self.output_path.stem}_contain_move_annotations.txt"), "w") as file:

            for vid_output_path in tqdm(self.video_scene_path.glob("*.json")):
                contain_and_move_frames = self._get_contain_move_mask(vid_output_path)
                contain_and_move_frames = list(map(str, contain_and_move_frames))
                contain_and_move_annot = ",".join(contain_and_move_frames)
                output_line = f"{vid_output_path.stem}\t{contain_and_move_annot}\n"
                file.write(output_line)

    def generate_snitch_visible_frames(self):
        self._init_files_if_not_initiated(with_scenes_data=True)

        with open(str(self.output_path / f"{self.output_path.stem}_snitch_visible_frames_{self.visible_ratio}.txt"), "w") as file:

            for vid_name, scene_data, gt_bb_data in tqdm(zip(self.video_names, self.scenes_data, self.gt_bb_data), total=len(self.video_names)):
                visible_snitch: List[bool] = []
                contain_list = self._get_objects_contained_frames(str(self.video_scene_path / vid_name) + ".json",
                                                                  self.class_names)
                uncontained_objs_data = self._get_uncontained_bb(contain_list, scene_data, gt_bb_data)

                all_frames_bb: List[List[np.ndarray]] = uncontained_objs_data["bb"]
                all_frames_labels: List[np.ndarray] = uncontained_objs_data["labels"]
                all_frames_coord: List[List[float]] = uncontained_objs_data["3d_coord"]

                for frame_bb, frame_label, frame_coord in zip(all_frames_bb, all_frames_labels, all_frames_coord):
                    if SNITCH_INDEX not in frame_label:
                        visible_snitch.append(False)

                    else:
                        snitch_visibility_flag = self._exclude_occluded_objects(frame_bb, frame_label, frame_coord, mode="snitch")
                        visible_snitch.append(snitch_visibility_flag)

                snitch_visible_frames: np.ndarray = np.where(visible_snitch)[0]
                snitch_visible_frames: List[str] = list(map(str, snitch_visible_frames))
                snitch_visible_annot: str = ",".join(snitch_visible_frames)
                output_line: str = f"{vid_name}\t{snitch_visible_annot}\n"
                file.write(output_line)


if __name__ == '__main__':

    vis_pred = VisibleObjectPredictions("samples.txt",
                                        "scenes",
                                        "labels",
                                        "data/containment_and_occlusions",
                                        visible_ratio=0.99)

    vis_pred.generate_snitch_visible_frames()



