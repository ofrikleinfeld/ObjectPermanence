import os
import os.path as osp
import glob
import errno

from tqdm import tqdm
import cv2


def mkdir_p(path):
    """
    Make all directories in `path`. Ignore errors if a directory exists.
    Equivalent to `mkdir -p` in the command line, hence the name.
    """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def check_for_corrupt_videos(vid_path: str = "the_path_to_your_videos",
                             save_file_name: str = "corrupt_videos.txt"):
    """

    Args:
        vid_path: (str) Path to folder which stores the generated videos
        save_file_name: (str) Name of saved txt file which contains the corrupt videos

    Returns: None

    """
    corrupt = []
    for vid in tqdm(glob.glob(osp.join(vid_path, '*.avi')), desc="Check for invalid videos"):
        cap = cv2.VideoCapture(vid)
        if not cap.isOpened():
            raise 'Unable to open video {}'.format(vid)

        flag, frame = cap.read()
        if frame is None:
            corrupt.append(osp.basename(vid).split(".")[0])
    if corrupt:
        with open(osp.join(osp.split(vid_path)[0], save_file_name), "w") as f:
            for item in corrupt:
                f.write("%s\n" % item)
