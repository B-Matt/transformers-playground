import torch
import numpy as np

from utils.dataset import Dataset
from typing import Tuple

from .engines.pytorch import PytorchEngine
from .engines.onnx import OnnxEngine
from .engines.tensorrt import TensorRTEngine

# Logging
from utils.logging import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class Prediction:
    def __init__(
        self,
        file_path: str,
        encoder: str,
        image_resolution: Tuple[int, int],
        channels,
        classes
    ) -> None:
        torch.cuda.empty_cache()
        self.image_resolution = image_resolution
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if file_path.endswith('.onnx'):
            self.engine = OnnxEngine(file_path, image_resolution)
        elif file_path.endswith('.engine'):
          self.engine = TensorRTEngine(file_path, image_resolution)
        else:
            self.engine = PytorchEngine(file_path, encoder, image_resolution, channels, classes)

    def warmup(self, warmup_iters: int) -> None:
        log.info(f'[PREDICTION]: Model warm up for {warmup_iters} iteration/s.')
        dummy_input = torch.rand((1, 3, self.image_resolution[0], self.image_resolution[1]))
        if isinstance(self.engine, PytorchEngine) or isinstance(self.engine, OnnxEngine):
            dummy_input = dummy_input.to(self.device)

        for i in range(1, warmup_iters):
            self.engine.predict_image(dummy_input, None)

    def predict_image(self, image: np.array, threshold: float = 0.5, resize: bool = False) -> np.array:
        # Resize image to preserve CUDA memory
        if resize:
            image = Dataset._resize_and_pad(image, (self.image_resolution[0], self.image_resolution[1]), (0, 0, 0))

        patch_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        if isinstance(self.engine, PytorchEngine) or isinstance(self.engine, OnnxEngine):
            patch_tensor = patch_tensor.to(self.device)

        return self.engine.predict_image(patch_tensor, threshold)
