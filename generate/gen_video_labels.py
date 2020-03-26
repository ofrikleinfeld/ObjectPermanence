import json
import argparse
from typing import List, Dict, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


SNITCH_NAME = "Spl_0"
SNITCH_LABEL = 140
NUM_FRAMES_ZERO_BASED = 299


def _parse_occlusions_file(file_path: str) -> Dict[str, List[int]]:
    results_dic: Dict[str, List[int]] = {}

    with open(file_path, "r") as all_f:
        for line in all_f:
            line = line[:-1]  # remove line ending
            vid_name, occ_frames_str = line.split("\t")

            if occ_frames_str == "":  # not occlusions at all in video
                results_dic[vid_name] = []
            else:
                occ_frames = [int(frame_index) for frame_index in occ_frames_str.split(",")]
                results_dic[vid_name] = occ_frames

    return results_dic


def _get_static_occlusion_frames(all_occlusions: List[int], moving_occlusions: List[int]) -> List[int]:
    static_occ_frames: List[int] = []

    all_occ_index = 0
    moving_occ_index = 0

    while all_occ_index < len(all_occlusions) and moving_occ_index < len(moving_occlusions):
        current_occ_frame = all_occlusions[all_occ_index]
        current_moving_occ_frame = moving_occlusions[moving_occ_index]

        if current_occ_frame != current_moving_occ_frame:
            static_occ_frames.append(current_occ_frame)
            all_occ_index += 1

        else:
            all_occ_index += 1
            moving_occ_index += 1

    if all_occ_index < len(all_occlusions):
        static_occ_frames.extend(all_occlusions[all_occ_index:])

    return static_occ_frames


def separate_containment_frame_to_static_and_moving(all_containment_annotations_path: str, ontainment_with_move_annotationspath: str, output_file_path: str) -> None:
    static_containment: Dict[str, List[int]] = {}

    # init all containment and moving containment from files
    all_containment: Dict[str, List[int]] = _parse_occlusions_file(all_containment_annotations_path)
    moving_containment: Dict[str, List[int]] = _parse_occlusions_file(ontainment_with_move_annotationspath)
    all_videos = sorted(all_containment.keys())

    # get only static occlusion frames
    for vid_name in tqdm(all_videos):
        video_all_containment = all_containment[vid_name]
        video_moving_containment = moving_containment.get(vid_name, None)

        # skip this file
        if video_moving_containment is None:
            continue

        # no occlusions frames at all
        if len(video_all_containment) == 0:
            static_containment_frames = []

        # all occlusions are static
        elif len(video_moving_containment) == 0:
            static_containment_frames = video_all_containment

        # some occlusions contain movements and some are not
        else:
            static_containment_frames = _get_static_occlusion_frames(video_all_containment, video_moving_containment)

        static_containment[vid_name] = static_containment_frames

    # write static occlusions to file
    with open(output_file_path, "w") as out_f:
        for vid_name, static_occ_frames in static_containment.items():
            static_occ_frames_str = ",".join([str(frame_index) for frame_index in static_occ_frames])
            video_line = f"{vid_name}\t{static_occ_frames_str}\n"
            out_f.write(video_line)


def get_snitch_containment_with_move_frames(scene_annotations: dict) -> List[int]:
    snitch_containment_with_move_ranges: List[List[int]] = []
    movements_annotations = scene_annotations["movements"]

    for obj_name, action_list in movements_annotations.items():

        contain_flag = ["_contain" in action[0] for action in action_list]
        pick_place_flag = ["_pick_place" in action[0] for action in action_list]
        slide_flag = ["_slide" in action[0] for action in action_list]

        if ("Cone" in obj_name) and (sum(contain_flag) != 0):
            # get the contain / pick place indexes
            for actions_index, containment_tag in enumerate(contain_flag):
                if containment_tag:  # it is indeed a containment
                    _, contained_object, _, containment_start = movements_annotations[obj_name][actions_index]

                    if contained_object == SNITCH_NAME:

                        next_slide_indexes = np.argwhere(slide_flag[actions_index:]).flatten() + actions_index
                        next_pick_place_indexes = np.argwhere(pick_place_flag[actions_index:]).flatten() + actions_index

                        if len(next_slide_indexes) > 0:  # otherwise there is no slide event at all after containment

                            if len(next_pick_place_indexes) > 0:
                                # the relevant pick place is the first pick place after the containment
                                relevant_pick_place_index = next_pick_place_indexes[0]
                                _, _, containment_end, _ = movements_annotations[obj_name][relevant_pick_place_index]
                            else:
                                containment_end = NUM_FRAMES_ZERO_BASED

                            # we have the end of containment frame
                            # every slide that happened before the end of containment frame is a valid slide
                            for slide_index in next_slide_indexes:
                                _, _, slide_start, slide_end = movements_annotations[obj_name][slide_index]
                                if slide_end <= containment_end:
                                    snitch_containment_with_move_ranges.append([slide_start, slide_end])

    # combine ranges to one list of frames
    snitch_containment_with_move_frames: List[int] = []
    for start_frame, end_frame in snitch_containment_with_move_ranges:
        containment_range = list(range(start_frame, end_frame + 1))
        snitch_containment_with_move_frames.extend(containment_range)

    snitch_containment_with_move_frames.sort()

    return snitch_containment_with_move_frames


def _cvt_obj_name_to_class(obj_name: str, video_data: dict) -> str:
    for obj in video_data["objects"]:
        if obj_name == obj["instance"]:
            return "_".join(obj[att] for att in ["size", "color", "shape", "material"])


def _cvt_class_to_label(obj_name, class_names_dict):
    return class_names_dict[obj_name]


def get_object_containment_frames(scene_annotations: dict, class_names_path: str, checked_object: str = SNITCH_NAME) -> Tuple[List[int], List[int], List[str]]:

    with open(class_names_path, "r") as file:
        class_names_dict: Dict[str, int] = json.load(file)

    object_containment_ranges: List[List[int]] = []
    containing_object: List[int] = []
    containing_object_names: List[str] = []
    movements_annotations = scene_annotations["movements"]

    for obj_name, action_list in movements_annotations.items():

        contain_flag = ["_contain" in action[0] for action in action_list]
        pick_place_flag = ["_pick_place" in action[0] for action in action_list]

        if ("Cone" in obj_name) and (sum(contain_flag) != 0):
            # get the contain / pick place indexes
            for actions_index, containment_tag in enumerate(contain_flag):
                if containment_tag:  # it is indeed a containment
                    _, contained_object, _, containment_start = movements_annotations[obj_name][actions_index]

                    if contained_object == checked_object:

                        next_pick_place_indexes = np.argwhere(pick_place_flag[actions_index:]).flatten() + actions_index
                        if len(next_pick_place_indexes) > 0:
                            # the relevant pick place is the first pick place after the containment
                            relevant_pick_place_index = next_pick_place_indexes[0]
                            _, _, containment_end, _ = movements_annotations[obj_name][relevant_pick_place_index]
                        else:
                            containment_end = NUM_FRAMES_ZERO_BASED

                        obj_class_name = _cvt_obj_name_to_class(obj_name, scene_annotations)
                        obj_label = _cvt_class_to_label(obj_class_name, class_names_dict)
                        object_containment_ranges.append([containment_start, containment_end])
                        containing_object.extend([obj_label] * (containment_end - containment_start + 1))
                        containing_object_names.append(obj_name)

    # combine ranges to one list of frames
    object_containment_frames: List[int] = []
    for start_frame, end_frame in object_containment_ranges:
        containment_range = list(range(start_frame, end_frame + 1))
        object_containment_frames.extend(containment_range)

    object_containment_frames.sort()

    return object_containment_frames, containing_object, containing_object_names


def get_tracked_object(scene_annotations: dict, class_json_path: str) -> Tuple[List[int], int]:
    babushka_count: int = 0
    tracked_object = SNITCH_LABEL * np.ones(NUM_FRAMES_ZERO_BASED+1, dtype=int)
    snitch_containment_frame_index, snitch_containing_objects_labels, snitch_containing_objects_names = get_object_containment_frames(scene_annotations, class_json_path)
    if len(snitch_containment_frame_index) != 0:
        tracked_object[snitch_containment_frame_index] = snitch_containing_objects_labels

        for object_name in set(snitch_containing_objects_names):
            containment_frame_index, containing_objects_labels, containing_objects_names = get_object_containment_frames(scene_annotations, class_json_path, checked_object=object_name)
            if len(containment_frame_index) != 0:
                tracked_object[containment_frame_index] = containing_objects_labels
                babushka_count = len(containment_frame_index)

    return tracked_object, babushka_count


def get_video_objects(scene_annotations: dict) -> List[str]:
    objects_and_locations: List[dict] = scene_annotations["objects"]
    video_objects_names = []
    for object_annotations in objects_and_locations:
        size = object_annotations["size"]
        material = object_annotations["material"]
        color = object_annotations["color"]
        shape = object_annotations["shape"]
        objects_names = f"{size}_{material}_{color}_{shape}"
        video_objects_names.append(objects_names)

    return video_objects_names


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create GT annotations for frames with snitch occlusions')
    subparsers = parser.add_subparsers()

    objects_parser = subparsers.add_parser('video_objects')
    objects_parser.set_defaults(mode='video_objects')
    objects_parser.add_argument("--annotations_dir", type=str, required=True, metavar='CATER/scenes',
                        help='Base directory containing json files with scene annotations for each video')
    objects_parser.add_argument("--output_file", type=str, required=True, metavar="output_annotations.json",
                        help="Path to save to output file containing GT occlusion annotations")

    snitch_containment_parser = subparsers.add_parser('snitch_containment')
    snitch_containment_parser.set_defaults(mode='snitch_containment')
    snitch_containment_parser.add_argument("--scenes_dir", type=str, required=True, metavar='CATER/scenes',
                        help='Base directory containing json files with scene annotations for each video')
    snitch_containment_parser.add_argument("--output_file", type=str, required=True, metavar="containment_annotations.txt",
                        help="Path to save to output file containing GT containment annotations for all videos")

    snitch_containment_parser = subparsers.add_parser('tracked_object')
    snitch_containment_parser.set_defaults(mode='tracked_object')
    snitch_containment_parser.add_argument("--scenes_dir", type=str, required=True, metavar='CATER/scenes',
                        help='Base directory containing json files with scene annotations for each video')
    snitch_containment_parser.add_argument("--output_dir", type=str, required=True, metavar="containment_annotations.txt",
                        help="Path to save to output dir containing GT containment annotations for all videos")
    snitch_containment_parser.add_argument("--class_names_path", type=str, required=True,
                                           help='Path to class_names json file - mapping from object name to label')

    snitch_containment_with_move_parser = subparsers.add_parser('snitch_containment_with_move')
    snitch_containment_with_move_parser.set_defaults(mode='snitch_containment_with_move')
    snitch_containment_with_move_parser.add_argument("--scenes_dir", type=str, required=True, metavar='CATER/scenes',
                        help='Base directory containing json files with scene annotations for each video')
    snitch_containment_with_move_parser.add_argument("--output_file", type=str, required=True, metavar="containment__with_move_annotations.txt",
                        help="Path to save to output file containing GT containment annotations for all videos")
    snitch_containment_with_move_parser.add_argument("--class_names_path", type=str, required=True,
                                           help='Path to class_names json file - mapping from object name to label')

    snitch_separate_containment_parser = subparsers.add_parser('separate_containment')
    snitch_separate_containment_parser.set_defaults(mode='separate_containment')
    snitch_separate_containment_parser.add_argument("--containment_annotations", type=str, required=True, metavar='occlusion_annotations.txt',
                                          help='Path to a file containing for each video list of occlusions frames')
    snitch_separate_containment_parser.add_argument("--containment_with_move_annotations", type=str, required=True, metavar='occlusion_with_movement_annotations.txt',
                                          help='Path to a file containing for each video list of occlusions with movements frames')
    snitch_separate_containment_parser.add_argument("--output_file", type=str, required=True, metavar="occlusions_only_static_annotations.txt",
                                            help="Path to save to output file containing only static occlusion annotations")

    args = parser.parse_args()
    mode = args.mode
    if mode == "snitch_containment" or mode == "snitch_containment_with_move" or mode == "tracked_object":
        scenes_dir = Path(args.scenes_dir)
        output_dir = args.output_dir
        class_names_path = args.class_names_path

        containment_frames: Dict[str, str] = {}
        count_babushka_frames: Dict[str, List[Any]] = {"video_name": [], "num_babushka_frames": []}

        videos_annotations: Dict[str, str] = {scene_path.stem: str(scene_path) for scene_path in scenes_dir.glob("*.json")}

        for video_name, scene_path in tqdm(videos_annotations.items()):
            with open(scene_path, "rb") as f:
                video_annotations = json.load(f)

            if mode == "snitch_containment":
                video_frames, _, _ = get_object_containment_frames(video_annotations, class_names_path)
            elif mode == "snitch_containment_with_move":
                video_frames: List[int] = get_snitch_containment_with_move_frames(video_annotations)
            else:
                video_frames, num_babushka_frames = get_tracked_object(video_annotations, class_names_path)
            video_containment_str = ",".join([str(frame) for frame in video_frames])
            containment_frames[video_name] = video_containment_str
            count_babushka_frames["video_name"].append(video_name)
            count_babushka_frames["num_babushka_frames"].append(num_babushka_frames)

        with open(str(Path(output_dir) / "tracked_object.txt"), "w") as f:
            output_lines = [f"{video}\t{frames}\n" for video, frames in containment_frames.items()]
            f.writelines(output_lines)

        df = pd.DataFrame(count_babushka_frames)
        df.to_csv(str(Path(output_dir) / "babushka.csv"), index=False)

    elif mode == "separate_containment":
        all_containment_path: str = args.containment_annotations
        moving_containment_path: str = args.containment_with_move_annotations
        output_path: str = args.output_file

        separate_containment_frame_to_static_and_moving(all_containment_path, moving_containment_path, output_path)

    else:
        raise NotImplementedError(f"{mode} mode is not supported")
