from typing import Dict, List

import numpy as np

from baselines.tracking_utils import ResultsAnalyzer


def cal_map(x: np.ndarray):
    return x.sum() / x.shape[0]


def analyze_results(predictions_dir: str, labels_dir: str, output_file: str,
                    containment_annotations: str, containment_only_static: str,
                    containment_with_movements: str, visibility_gt_0: str, visibility_gt_30: str,
                    visibility_gt_99: str, iou_thresh: List[float]):
    analyzer: ResultsAnalyzer = ResultsAnalyzer.init_from_files(predictions_dir, labels_dir, iou_thresh)
    for metric, agg_func in zip(["iou", "map"], [np.mean, cal_map]):
        analyzer.compute_aggregated_metric(aggregations_name="overall", aggregation_function=agg_func, metric=metric)

        if containment_annotations is not None:
            containment_mask: Dict[str, np.ndarray] = analyzer.get_frames_mask(containment_annotations)
            analyzer.compute_aggregated_metric_masking_frames("contained", agg_func, containment_mask, metric=metric)

        if containment_only_static is not None:
            containment_static_mask: Dict[str, np.ndarray] = analyzer.get_frames_mask(containment_only_static)
            analyzer.compute_aggregated_metric_masking_frames("static_contained", agg_func, containment_static_mask, metric=metric)

        if containment_with_movements is not None:
            containment_with_move_mask = analyzer.get_frames_mask(containment_with_movements)
            analyzer.compute_aggregated_metric_masking_frames("contained_with_move", agg_func, containment_with_move_mask, metric=metric)

        if visibility_gt_0 is not None:
            visibility_gt_0_mask = analyzer.get_frames_mask(visibility_gt_0)
            analyzer.compute_aggregated_metric_masking_frames("visibility_gt_0", agg_func, visibility_gt_0_mask, metric=metric)

            if containment_annotations is not None:
                # full occlusions == not visible at all and not contained
                not_visible_at_all_mask = {video: ~mask for video, mask in visibility_gt_0_mask.items()}
                not_visible_not_contained_mask = {video: np.logical_and(not_visible_mask, ~containment_mask[video])
                                                  for video, not_visible_mask, in not_visible_at_all_mask.items()}
                analyzer.compute_aggregated_metric_masking_frames("full_occlusion", agg_func, not_visible_not_contained_mask, metric=metric)

        if visibility_gt_30 is not None:
            visibility_gt_30_mask = analyzer.get_frames_mask(visibility_gt_30)
            analyzer.compute_aggregated_metric_masking_frames("visibility_gt_30", agg_func, visibility_gt_30_mask, metric=metric)

        if visibility_gt_99 is not None:
            visibility_gt_99_mask = analyzer.get_frames_mask(visibility_gt_99)
            analyzer.compute_aggregated_metric_masking_frames("visibility_gt_99", agg_func, visibility_gt_99_mask, metric=metric)

    analyzer.write_results(output_file)
