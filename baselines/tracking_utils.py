import json
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np
import cv2


class VideoHandling(object):
    def __init__(self, vid_path: str, output_path: str = None):
        self.vid_path = vid_path
        self.output_path = output_path
        self.cap: cv2.VideoCapture = None
        self.vid_writer: cv2.VideoWriter = None
        self.current_frame: np.ndarray = None
        self.current_frame_index: int = -1
        self.video_still_active: bool = False
        self.num_valid_frames = None

        self._init_video_cap()

    def _init_video_cap(self) -> None:
        self.cap = cv2.VideoCapture(self.vid_path)
        if not self.cap.isOpened():
            raise 'Unable to open video {}'.format(self.vid_path)
        # for some reason cap always returns extra frame
        # our labels are aligned with the 300 first frames
        # thus, we will omit the last frame that cv2 reader returns
        self.num_valid_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1

    def _init_video_writer(self, w_frame, h_frame) -> None:
        self.vid_writer = cv2.VideoWriter(self.output_path,
                                          cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                          30,
                                          (w_frame, h_frame))

    def get_current_frame(self) -> np.ndarray:
        return self.current_frame

    def get_current_frame_index(self) -> int:
        return self.current_frame_index

    def check_video_still_active(self) -> bool:
        return self.current_frame_index < self.num_valid_frames

    def write_debug_frame(self) -> None:
        frame = self.current_frame
        h_frame, w_frame, _ = frame.shape

        # init video writer if not initiated yet
        if self.vid_writer is None:
            self._init_video_writer(w_frame, h_frame)

        self.vid_writer.write(frame)

    def read_next_frame(self) -> None:
        _, frame = self.cap.read()

        # update current frame and status
        self.current_frame = frame
        self.current_frame_index += 1

    def write_bb_to_frame(self, bbox: List[int], color: Tuple[int, int, int]) -> None:
        frame = self.current_frame
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)

    def complete_video_writing(self) -> None:
        self.cap.release()
        self.vid_writer.release()


class DataHelper(object):

    @staticmethod
    def parse_obj_gt_bb(bb_gt: Dict[str, List[List[int]]], object_name: str = "small_gold_spl_metal_Spl_0") -> List[List[int]]:
        object_bb = bb_gt[object_name]

        # transform from xy_wh bb format to xy_xy format
        object_bb = [[x, y, x + w, y + h] for x, y, w, h in object_bb]

        return object_bb

    @staticmethod
    def read_obj_gt_bb(vid_path: str, bb_dir_path: str) -> List[List[int]]:
        video_name = Path(vid_path).stem
        video_bb_file_suffix = video_name + "_bb.json"
        video_bb_full_path = Path(bb_dir_path) / video_bb_file_suffix
        with open(video_bb_full_path, "rb") as f:
            bb_objects_gt: Dict[str, List[List[int]]] = json.load(f)
            snitch_gt_bb = DataHelper.parse_obj_gt_bb(bb_objects_gt)

            return snitch_gt_bb

    @staticmethod
    def write_bb_predictions_to_file(video_path: str, predictions_dir: str, snitch_bb_prediction: List[List[int]]) -> None:
        video_name = Path(video_path).stem
        prediction_file_name = video_name + "_bb.json"
        predictions_path = Path(predictions_dir) / prediction_file_name

        snitch_bb_prediction = [[int(x1), int(y1), int(x2), int(y2)] for [x1, y1, x2, y2] in snitch_bb_prediction]
        with open(predictions_path, 'w') as f:
            json.dump(snitch_bb_prediction, f, indent=2)


class ResultsAnalyzer(object):

    @staticmethod
    def compute_iou_for_frame(box_1: List[int], box_2: List[int]) -> float:
        """
        computes intersection over union of two bounding boxes arrays
        it's a symmetrical measurement, it doesn't matter which one it prediction
        and which one it ground truth
        """

        # obtain the X,Y of the intersection rectangle
        intersection_x_1: int = max(box_1[0], box_2[0])
        intersection_y_1: int = max(box_1[1], box_2[1])
        intersection_x_2: int = min(box_1[2], box_2[2])
        intersection_y_2: int = min(box_1[3], box_2[3])

        # calc the area of intersection rectangle
        intersection_area: int = max(0, intersection_x_2 - intersection_x_1 + 1) * max(0, intersection_y_2 - intersection_y_1 + 1)

        # compute the area of both the prediction and ground-truth rectangles
        box_1_area: int = (box_1[2] - box_1[0] + 1) * (box_1[3] - box_1[1] + 1)
        box_2_area: int = (box_2[2] - box_2[0] + 1) * (box_2[3] - box_2[1] + 1)

        # Subtracting interArea because we sum it twice
        union_area: int = box_1_area + box_2_area - intersection_area

        # compute intersection over union
        iou: float = intersection_area / float(union_area)

        return iou

    @staticmethod
    def compute_vectorized_iou_for_video(boxes_1: np.ndarray, boxes_2: np.ndarray) -> np.ndarray:

        # divide the boxes to coordinates
        x11, y11, x12, y12 = np.split(boxes_1, 4, axis=1)
        x21, y21, x22, y22 = np.split(boxes_2, 4, axis=1)

        # obtain XY of intersection rectangle area
        xA: np.ndarray = np.maximum(x11, x21)
        yA: np.ndarray = np.maximum(y11, y21)
        xB: np.ndarray = np.minimum(x12, x22)
        yB: np.ndarray = np.minimum(y12, y22)

        # compute intersection area
        interArea: np.ndarray = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)

        # compute each one of the boxes area
        boxAArea: np.ndarray = (x12 - x11 + 1) * (y12 - y11 + 1)
        boxBArea: np.ndarray = (x22 - x21 + 1) * (y22 - y21 + 1)

        iou: np.ndarray = interArea / (boxAArea + boxBArea - interArea)
        iou = iou.flatten()
        return iou

    @classmethod
    def init_from_files(cls, bb_prediction_dir: str, bb_gt_dir: str, iou_thresh: float = None):
        video_files: List[str] = []
        video_bb_predictions: Dict[str, List[List[int]]] = {}
        video_bb_gt: Dict[str, List[List[int]]] = {}

        # parse predictions
        predictions_files = Path(bb_prediction_dir).glob("*.json")
        for f_predict in predictions_files:
            # extract video name
            video_name = f_predict.stem[:-3]  # remove _bb suffix
            video_files.append(video_name)

            # read and parse json file
            with open(f_predict, "rb") as f:
                snitch_predictions_locations: List[List[int]] = json.load(f)
                # fix bug in current ground truth locations, remove first frame annotations
                # snitch_predictions_locations = snitch_predictions_locations[1:]
                video_bb_predictions[video_name] = snitch_predictions_locations

        # parse ground truth locations
        gt_files = Path(bb_gt_dir).glob("*.json")
        for f_gt in gt_files:
            # extract video name
            video_name = f_gt.stem[:-3]  # remove _bb suffix
            # skip videos we don't have predictions for
            if video_name in video_files:

                # read and parse json file
                with open(f_gt, "rb") as f:
                    all_objects_locations: Dict[str, List[List[int]]] = json.load(f)
                    snitch_gt_locations: List[List[int]] = DataHelper.parse_obj_gt_bb(all_objects_locations)

                    video_bb_gt[video_name] = snitch_gt_locations

        # sort data to make sure predictions and ground truth are aligned to same video
        sorted_bb_predictions = sorted(video_bb_predictions.items(), key=lambda x: x[0])
        sorted_bb_gt = sorted(video_bb_gt.items(), key=lambda x: x[0])

        video_files = sorted(video_files)
        bb_predictions: List[List[List[int]]] = [bb_pred for video_name, bb_pred in sorted_bb_predictions]
        bb_gt: List[List[List[int]]] = [bb_gt for video_name, bb_gt in sorted_bb_gt]

        return cls(video_files, bb_predictions, bb_gt, iou_thresh)

    def __init__(self, videos_files: List[str], bb_predictions: List[List[List[int]]], bb_gt: List[List[List[int]]], iou_thresh: List[float] = None):
        assert len(videos_files) == len(bb_predictions) == len(bb_gt)

        self.videos_names: List[str] = []
        self.videos_num_frames: Dict[str, int] = {}
        self.bb_predictions: Dict[str, List[List[int]]] = {}
        self.bb_gt: Dict[str, List[List[int]]] = {}

        self.iou_results: Dict[str, List[float]] = {}
        self.map_results: Dict[str, Dict[str, List[bool]]] = {thresh: {} for thresh in iou_thresh} if iou_thresh else {}
        self.videos_metrics: Dict[str, Dict[str, float]] = {}

        self.iou_thresh: List[float] = iou_thresh

        self._init_videos_data(videos_files, bb_predictions, bb_gt)
        self._compute_iou_results()
        self._compute_bool_overlap(self.iou_thresh)

    def _init_videos_data(self, videos_files: List[str], bb_predictions: List[List[List[int]]], bb_gt: List[List[List[int]]]) -> None:
        all_videos_names = list(map(lambda x: Path(x).stem, videos_files))
        num_potential_videos = len(all_videos_names)

        for i in range(num_potential_videos):
            current_video = all_videos_names[i]
            current_predictions = bb_predictions[i]
            current_gt = bb_gt[i]
            current_video_length = len(current_gt)

            if -100 in current_predictions:
                continue  # skip defected videos

            self.bb_predictions[current_video] = current_predictions
            self.bb_gt[current_video] = current_gt
            self.videos_num_frames[current_video] = current_video_length
            self.videos_names.append(current_video)

    def _compute_iou_results(self) -> None:

        for video_name in self.videos_names:
            video_predictions: np.ndarray = np.array(self.bb_predictions[video_name])
            video_gt: np.ndarray = np.array(self.bb_gt[video_name])

            batch_video_iou_results = self.compute_vectorized_iou_for_video(video_predictions, video_gt)
            self.iou_results[video_name] = batch_video_iou_results

    def _compute_bool_overlap(self, iou_thresh) -> None:
        if iou_thresh is not None:

            for threshold in iou_thresh:
                for video_name, frame_iou in self.iou_results.items():
                    self.map_results[threshold][video_name] = frame_iou > threshold

    def get_frames_mask(self, occlusion_frames_file: str) -> Dict[str, np.ndarray]:
        occlusion_masks = {}
        with open(occlusion_frames_file, "r") as f:
            for line in f:
                line = line[:-1]
                video_name, occlusion_frames_str = line.split("\t")
                if video_name not in self.bb_gt:
                    continue

                if occlusion_frames_str == "":
                    occlusion_frames = []
                else:
                    occlusion_frames = np.array(occlusion_frames_str.split(","), dtype=np.int)
                video_length = self.videos_num_frames[video_name]
                mask = np.zeros(video_length, dtype=np.bool)
                mask[occlusion_frames] = True
                occlusion_masks[video_name] = mask

        return occlusion_masks

    def compute_aggregated_metric(self, aggregations_name: str, aggregation_function, metric: str = "iou") -> None:

        if metric == "iou":
            video_mean_results = {}

            for video_name in self.videos_names:
                video_metric_results = np.array(self.iou_results[video_name])
                video_mean_metric = float(aggregation_function(video_metric_results))
                video_mean_results[video_name] = video_mean_metric

            self.videos_metrics[f"{aggregations_name}_{metric}"] = video_mean_results

        elif metric == "map":

            for iou_threshold, video_overlap_dict in self.map_results.items():
                video_mean_results = {}

                for video_name in self.videos_names:
                    video_metric_results = np.array(video_overlap_dict[video_name])
                    video_mean_metric = float(aggregation_function(video_metric_results))
                    video_mean_results[video_name] = video_mean_metric

                self.videos_metrics[f"{aggregations_name}_{metric}_{iou_threshold}"] = video_mean_results

        else:
            raise NotImplementedError("This metric is not supported")

    def get_videos_names(self):
        return self.videos_names

    def compute_metric_mask(self, occlusions_mask: Dict[str, np.ndarray], video_metric_results: np.ndarray, aggregation_function, video_name: str):
        video_mean_results = {}
        video_mask: List[bool] = occlusions_mask[video_name]
        num_mask_frames = np.sum(video_mask)

        if num_mask_frames == 0:
            video_mean_metric = np.nan
        else:
            video_metric_mask_frames = video_metric_results[video_mask]
            video_mean_metric = float(aggregation_function(video_metric_mask_frames))

        video_mean_results[video_name] = video_mean_metric

        return video_mean_results

    def compute_aggregated_metric_masking_frames(self, aggregation_name: str, aggregation_function, occlusions_mask: Dict[str, np.ndarray], metric: str = "iou") -> None:
        ratio_column_name = f"{aggregation_name}_ratio"
        video_mask_ratio = {}

        if metric == "iou":
            video_metric = {}
            metric_column_name = f"{aggregation_name}_mean_{metric}"
            for video_name in self.videos_names:
                video_metric_results = np.array(self.iou_results[video_name])
                video_metric.update(self.compute_metric_mask(occlusions_mask, video_metric_results,
                                                             aggregation_function, video_name))
                video_mask = occlusions_mask[video_name]
                num_mask_frames = np.sum(video_mask)

                if num_mask_frames == 0:
                    mask_ratio = 0.0
                else:
                    mask_ratio = num_mask_frames / len(video_mask)

                video_mask_ratio[video_name] = mask_ratio

            self.videos_metrics[metric_column_name] = video_metric
            self.videos_metrics[ratio_column_name] = video_mask_ratio

        elif metric == "map":
            video_metric = {key: {} for key in self.map_results.keys()}
            for video_name in self.videos_names:
                for threshold, video_overlap in self.map_results.items():
                    video_metric_results = np.array(video_overlap[video_name])
                    video_metric[threshold].update(self.compute_metric_mask(occlusions_mask, video_metric_results,
                                                                            aggregation_function, video_name))
            for threshold, map_value in video_metric.items():
                metric_column_name = f"{aggregation_name}_mean_{metric}_{threshold}"
                self.videos_metrics[metric_column_name] = map_value
        else:
            raise NotImplementedError("This metric is not supported")

    def compute_precision_data(self, thresholds: List[float] = None, occlusions_mask: Dict[str, np.ndarray] = None):
        if thresholds is None:
            thresholds = [i / 20 for i in range(20)]  # 0 - 0.95 with 0.05 step

        with_occlusions = False
        if occlusions_mask is not None:
            with_occlusions = True

        for t in thresholds:
            def t_agg_func(x):
                return np.sum(x > t) / x.shape[0]

            if with_occlusions:
                aggregation_name = f"occ_precision_{t}"
                self.compute_aggregated_metric_masking_frames("iou", aggregation_name, t_agg_func, occlusions_mask)
            else:
                aggregation_name = f"precision_{t}"
                self.compute_aggregated_metric("iou", aggregation_name, t_agg_func)

    def get_analysis_df(self) -> pd.DataFrame:
        video_data = {
            "videos_names": sorted(self.videos_names)
        }

        for metric_name, metric_data_dict in self.videos_metrics.items():
            sorted_metric_data = sorted(metric_data_dict.items(), key=lambda x: x[0])
            metric_values = list(map(lambda x: x[1], sorted_metric_data))
            video_data[metric_name] = metric_values

        results_df = pd.DataFrame.from_dict(video_data)
        return results_df

    def write_results(self, results_filepath: str) -> None:
        # create a dictionary later be converted to DataFrame

        results_df = self.get_analysis_df()
        results_df = results_df.round(3)
        results_df.to_csv(results_filepath, index=None)

