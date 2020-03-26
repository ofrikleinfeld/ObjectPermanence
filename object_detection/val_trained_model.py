from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader

import object_detection.utils as utils
from object_detection.datasets import CaterObjectDetectionDataset
from object_detection.models import get_fast_rcnn_for_fine_tune
from object_indices import OBJECTS_NAME_TO_IDX


BASE_DIR = "./"


def validate_bb(model, data_loader, class_names, thresh=0.6):
    with torch.no_grad():
        model.eval()
        inputs, y = next(iter(data_loader))
        y_tag = model(inputs)
    acc_boxes = [(output['scores'] > thresh).tolist() for output in y_tag]
    # y_tag_acc = y_tag[y_tag]

    for i,(image, result) in enumerate(zip(inputs, y_tag)):
        detected_label = [list(class_names.keys())[label] for label in y_tag[i]['labels'][acc_boxes[i]]]
        image = image.to('cpu').transpose(0, 2).transpose(0, 1).numpy()
        for y_,bb in zip(detected_label, result['boxes'][acc_boxes[i]]):
            bb = bb.to('cpu').numpy().astype('int32')
            image = cv2.rectangle(image, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 0), 1)
            cv2.putText(image, y_, (bb[0], bb[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (36, 255, 12), 1)
        cv2.imwrite(f"{BASE_DIR}/{i}.png", image)
        # cv2.imshow('output', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # cv2.waitKey(0)


if __name__ == '__main__':
    root_dir = Path(BASE_DIR) / "object_detection"
    data_path = root_dir / "OD_data"
    train_file_names_path = root_dir / "od_train_filenames.txt"
    train_labels_path = root_dir / "train.csv"
    validation_file_names_path = root_dir / "od_val_filenames.txt"
    validation_labels_path = root_dir / "validation.csv"
    save_path = Path(BASE_DIR) / "saved_detection_models"
    # number of classes is known and predefined according to the CATER dataset characteristics
    num_classes = 193
    batch_size = 8

    model: torch.nn.Module = get_fast_rcnn_for_fine_tune(num_classes)
    checkpoint = torch.load(save_path / "detection_model_epoch_0_iter_7850.pth")
    model.load_state_dict(checkpoint["model_state_dict"])

    dataset_train = CaterObjectDetectionDataset(data_path, train_file_names_path, train_labels_path)
    dataset_validation = CaterObjectDetectionDataset(data_path, validation_file_names_path, validation_labels_path)

    # define training and validation data loaders
    train_loader = DataLoader(dataset_train, batch_size=8, shuffle=True, num_workers=12, collate_fn=utils.collate_fn)
    dev_loader = DataLoader(dataset_validation, batch_size=8, shuffle=True, num_workers=12, collate_fn=utils.collate_fn)
    class_names = OBJECTS_NAME_TO_IDX

    validate_bb(model, dev_loader, class_names)

