import os
import os.path as osp
import json
import pickle
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from object_indices import OBJECTS_NAME_TO_IDX


class CaterObjectDetectionDataset(Dataset):

    def __init__(self, data_path: str, file_names_path: str, labels_path: str):
        self.data_path = Path(data_path)
        self.file_names_path = file_names_path
        self.labels_path = labels_path
        self.file_names: List[str] = None
        self.labels_data_frame: pd.DataFrame = None
        self.object_labels_to_indices: dict = OBJECTS_NAME_TO_IDX

    def _load_data_files_if_not_loaded(self) -> None:

        if self.file_names is None:

            # read images names
            with open(self.file_names_path, "r") as f:
                self.file_names = f.read().splitlines()

            # read labels data
            self.labels_data_frame = pd.read_csv(self.labels_path)

    def __len__(self) -> int:
        self._load_data_files_if_not_loaded()
        return len(self.file_names)

    def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.tensor]:
        self._load_data_files_if_not_loaded()

        # load image
        img_name = self.file_names[idx]
        img_path: str = self.data_path / img_name
        img = Image.open(img_path).convert("RGB")

        # load labels
        img_df: pd.DataFrame = self.labels_data_frame[self.labels_data_frame["filename"] == img_name]
        num_objects: int = len(img_df)
        objects_labels: np.ndarray = np.array(img_df["object_class"].apply(lambda x: self.object_labels_to_indices[x]))
        objects_areas: np.ndarray = np.array(img_df["width"] * img_df["height"])
        boxes: List[np.ndarray] = [
            np.array(img_df["X"]),
            np.array(img_df["Y"]),
            np.array(img_df["X"] + img_df["width"]),
            np.array(img_df["Y"] + img_df["height"])
        ]

        # transform to tensors and add other labels data
        boxes_tensor: torch.tensor = torch.as_tensor(boxes, dtype=torch.float32).transpose(1, 0)  # transpose to shape (num objects, 4)
        labels_tensor: torch.tensor = torch.as_tensor(objects_labels, dtype=torch.int64)
        areas_tensor: torch.tensor = torch.as_tensor(objects_areas, dtype=torch.float32)
        image_id_tensor: torch.tensor = torch.as_tensor(idx, dtype=torch.int64)
        # image_id_tensor: torch.tensor = idx * torch.ones(num_objects, dtype=torch.int64)
        is_crowded_tensor: torch.tensor = torch.zeros(num_objects, dtype=torch.int64)

        # apply transformation
        # return image and labels data
        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": image_id_tensor,
            "area": areas_tensor,
            "iscrowd": is_crowded_tensor
        }

        img = T.ToTensor()(img)
        return img, target


class PennFudanDataset(Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class CaterBaselineDataset(Dataset):
    def __init__(self, pred_file_path:str, gt_file_path:str, file_names_path:str, class_names_path:str):
        self.pred_file_path = pred_file_path
        self.gt_file_path = gt_file_path
        self.file_names_path = file_names_path
        self.class_names_path = class_names_path
        self.file_names = None
        self.class_names_dict = None
        self.gt_class_indexes = None

    def _load_data_files_if_not_loaded(self) -> None:

        if self.file_names is None:

            # read images names
            with open(self.file_names_path, "r") as f:
                self.file_names = f.read().splitlines()

            self.class_names_dict = self._load_json(self.class_names_path)

    def _load_pkl(self, path) -> dict:
        with open(path,'rb') as file:
            data = pickle.load(file)
        return data

    def _load_json(self, path) -> dict:
        with open(path,'rb') as file:
            data = json.load(file)
        return data

    def _pad_pred_data(self, bb_list:list, label_list:list):
        missing_obj = set(label_list) - set(self.gt_class_indexes)
        if missing_obj:
            bboxes = [-1*np.ones(4) for _ in range(len(missing_obj))]
        return bb_list.append(bboxes), label_list.append(missing_obj)

    def __len__(self) -> int:
        self._load_data_files_if_not_loaded()
        return len(self.file_names)

    def __getitem__(self, idx):
        vid_name = self.file_names[idx]
        video_pred_file = osp.join(self.pred_file_path, vid_name+"_predictions.pkl")
        video_gt_file = osp.join(self.pred_file_path, vid_name+"_bb.json")
        pred_data = self._load_pkl(video_pred_file)
        gt_data = self._load_json(video_gt_file)
        # self.num_objects = len(gt_data.keys())
        self.gt_class_indexes = list(map(lambda x: self.class_names_dict[x], gt_data.keys()))
        pred_data = dict(map(lambda bb,lbl: self._pad_pred_data(bb,lbl), pred_data.items()))

