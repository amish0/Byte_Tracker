from pathlib import Path
import numpy as np

from .byte_tracker import BYTETracker
from .loadyaml import load_yaml
# TRACKER_MAP = {'bytetrack': BYTETracker, 'botsort': BOTSORT}

TRACKER_MAP = {'bytetrack': BYTETracker}
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory

class Tracker:
    """Tracker class for multi object tracking with some modifications to the original Sort class"""

    def __init__(self, tracker_type: str = 'sort') -> None:
        """@brief Tracker class
           @details will initialize the tracker with the given tracker_type and tracker parameters from corresponding yaml file
           @param tracker_type type of tracker to be used
        """
        if isinstance(tracker_type, str) and tracker_type in TRACKER_MAP.keys():
            self.parm = load_yaml(ROOT / "cfg/bytetrack.yaml")
            self.tracker = TRACKER_MAP[tracker_type](**self.parm)
        else:
            print("tracker_type must be string and in {}".format(TRACKER_MAP.keys()))

    def update(self, detections: dict, img: np.ndarray)->any:
        """@brief update the tracker with new detections
           @details update the tracker with new detections
           @param detections new detections in the format [[x1, y1, x2, y2, score, class_id], ...]
           @return updated bounding boxes in the format [[x1, y1, x2, y2, score, class_id, track_id], ...] if tracker is not initalized it will return None
        """
        if hasattr(self, 'tracker'):
            return self.tracker.update(detections, img)
        else:
            print("tracker not init")
            return None

    def __call__(self, detections, img):
        """@brief update the tracker with new detections
           @details update the tracker with new detections
           @param detections new detections in the format [[x1, y1, x2, y2, score, class_id], ...]
           @return updated bounding boxes in the format [[x1, y1, x2, y2, score, class_id, track_id], ...] if tracker is not initalized it will return None"""
        return self.update(detections, img)

    def check_parameters(self):
        print(self.parm)