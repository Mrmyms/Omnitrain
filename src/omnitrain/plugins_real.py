import os
import numpy as np
import pandas as pd
from typing import Any, Optional
from .plugins import ModalityPlugin


class CSVModalityPlugin(ModalityPlugin):
    """Plugin that ingests tabular data from a CSV file as sensor tokens."""

    def __init__(self, bus, modal_id, frequency_hz, write_ptr, csv_path, feature_cols=None):
        super().__init__(bus, modal_id, frequency_hz, write_ptr=write_ptr)
        self.df = pd.read_csv(csv_path)
        self.idx = 0
        self.cols = feature_cols or self.df.columns.tolist()

    def read_raw_data(self) -> Any:
        # Loop over data for simulation
        row = self.df.iloc[self.idx % len(self.df)][self.cols].values
        self.idx += 1
        return row

    def encode(self, raw_data: Any) -> np.ndarray:
        base = np.zeros(512, dtype='float32')
        base[:min(512, len(raw_data))] = raw_data[:512]
        return base


class ImageFolderPlugin(ModalityPlugin):
    """Plugin that ingests images from a directory as flattened sensor tokens."""

    def __init__(self, bus, modal_id, frequency_hz, write_ptr, img_dir):
        super().__init__(bus, modal_id, frequency_hz, write_ptr=write_ptr)
        self.img_dir = img_dir
        self.images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg'))]
        self.idx = 0

    def read_raw_data(self) -> Any:
        # Lazy cv2 import for robust file reading
        try:
            import cv2
        except ImportError:
            raise ImportError("opencv-python required: pip install opencv-python")

        if not self.images:
            return None
        path = os.path.join(self.img_dir, self.images[self.idx % len(self.images)])
        self.idx += 1

        img = cv2.imread(path)
        if img is None:
            print(f"[ImagePlugin:{self.modal_id}] WARN: Could not load {path}")
        return img

    def encode(self, raw_data: Any) -> np.ndarray:
        if raw_data is None:
            return np.zeros(512, dtype='float32')
        import cv2
        resized = cv2.resize(raw_data, (16, 32)).flatten()[:512]
        token = np.zeros(512, dtype='float32')
        token[:len(resized)] = resized
        return token
