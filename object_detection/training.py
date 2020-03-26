from pathlib import Path

import torch
from torch.utils.data import DataLoader

from object_detection.datasets import CaterObjectDetectionDataset
from object_detection.models import get_fast_rcnn_for_fine_tune
from object_detection.engine import train_one_epoch, evaluate
from object_detection import utils

from tqdm import tqdm


if __name__ == '__main__':
    root_dir = Path("./object_detection")
    data_path = root_dir / "OD_data"
    train_file_names_path = root_dir / "od_train_filenames.txt"
    train_labels_path = root_dir / "train.csv"
    validation_file_names_path = root_dir / "od_val_filenames.txt"
    validation_labels_path = root_dir / "validation.csv"
    save_path = root_dir / "saved_detection_models"
    # number of classes is known and predefined according to the CATER dataset characteristics
    num_classes = 193

    # use the Cater dataset object
    dataset_train = CaterObjectDetectionDataset(data_path, train_file_names_path, train_labels_path)
    dataset_validation = CaterObjectDetectionDataset(data_path, validation_file_names_path, validation_labels_path)

    # define training and validation data loaders
    train_loader = DataLoader(dataset_train, batch_size=2, shuffle=True, num_workers=12, collate_fn=utils.collate_fn)
    dev_loader = DataLoader(dataset_validation, batch_size=2, shuffle=True, num_workers=12, collate_fn=utils.collate_fn)

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')

    # load saved model
    # checkpoint = torch.load(save_path / "detection_model_epoch_0_iter_900.pth")

    # get the model using our helper function
    model: torch.nn.Module = get_fast_rcnn_for_fine_tune(num_classes)

    # model.load_state_dict(checkpoint["model_state_dict"])

    # duplicate the model to all available gpus
    # model = torch.nn.DataParallel(model)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # define number of epochs
    num_epochs = 40
    print_freq = 100
    save_every = 100

    for epoch in tqdm(range(num_epochs)):

        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=200, save_path=save_path, save_every=save_every)

        # update the learning rate
        # lr_scheduler.step()

        # evaluate on the test dataset
        evaluate(model, dev_loader, device=device)

    print("That's it!")
