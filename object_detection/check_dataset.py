from pathlib import Path

from datasets import CaterObjectDetectionDataset

if __name__ == '__main__':
    root_dir = Path("./object_detection")
    data_path = root_dir / "OD_data"
    file_names_path = root_dir / "od_train_filenames.txt"
    labels_path = root_dir / "train.csv"
    cater_dataset = CaterObjectDetectionDataset(data_path, file_names_path, labels_path)

    for i in range(3):
        print(f"Sample number {i}")
        x, y = cater_dataset[i]
        print(f" X shape {x.size()}")
        for name, t in y.items():
            print(f" target parameter name: {name}, and shape {t.size()}")
