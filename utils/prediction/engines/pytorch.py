import time
import torch
import pathlib

import segmentation_models_pytorch as smp

from typing import Tuple
from utils.logging import logging

# Logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Engine Class
class PytorchEngine:
    def __init__(
        self, 
        model_path: str,
        encoder: str,
        image_resolution: Tuple[int, int],
        channels,
        classes
    ) -> None:
        log.info(f'[ENGINE]: Started PyTorch engine loading!')
        self.image_resolution = image_resolution
        self.load_model(model_path, encoder, channels, classes)
        log.info("[ENGINE]: PyTorch engine inference object initialized.")

    def load_model(self, model_path, encoder=None, n_channels=3, n_classes=2) -> None:
        log.info(f'[PYTORCH]: Loading model {model_path} ({encoder})')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        state_dict = torch.load(model_path, map_location=self.device)
        if state_dict['model_name'] == 'UnetPlusPlus':
            self.net = smp.UnetPlusPlus(encoder_name=(encoder if encoder else "resnet34"), encoder_weights="imagenet", decoder_use_batchnorm=True, in_channels=n_channels, classes=n_classes)
        elif state_dict['model_name'] == 'MAnet':
            self.net = smp.MAnet(encoder_name=(encoder if encoder else "resnet34"), encoder_depth=5, encoder_weights='imagenet', decoder_use_batchnorm=True, in_channels=n_channels, classes=n_classes)
        elif state_dict['model_name'] == 'Linknet':
            self.net = smp.Linknet(encoder_name=(encoder if encoder else "resnet34"), encoder_depth=5, encoder_weights="imagenet", decoder_use_batchnorm=True, in_channels=n_channels, classes=n_classes)
        elif state_dict['model_name'] == 'FPN':
            self.net = smp.FPN(encoder_name=(encoder if encoder else "resnet34"), encoder_weights="imagenet", in_channels=n_channels, classes=n_classes)
        elif state_dict['model_name'] == 'PSPNet':
            self.net = smp.PSPNet(encoder_name=(encoder if encoder else "resnet34"), encoder_weights="imagenet", in_channels=n_channels, classes=n_classes)
        elif state_dict['model_name'] == 'PAN':
            self.net = smp.PAN(encoder_name=(encoder if encoder else "resnet34"), encoder_weights="imagenet", in_channels=n_channels, classes=n_classes)
        elif state_dict['model_name'] == 'DeepLabV3':
            self.net = smp.DeepLabV3(encoder_name=(encoder if encoder else "resnet34"), encoder_weights="imagenet", in_channels=n_channels, classes=n_classes)
        elif state_dict['model_name'] == 'DeepLabV3Plus':
            self.net = smp.DeepLabV3Plus(encoder_name=(encoder if encoder else "resnet34"), encoder_weights="imagenet", in_channels=n_channels, classes=n_classes)
        else:
            self.net = smp.Unet(encoder_name=(encoder if encoder else "resnet34"), decoder_use_batchnorm=True, in_channels=n_channels, classes=n_classes)

        self.net.load_state_dict(state_dict['model_state'])
        self.net.to(self.device)
        self.net.eval()

    def predict_image(
        self,
        input_tensor: torch.Tensor,
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, int]:
        input_tensor = input_tensor.to(self.device)

        with torch.autocast(device_type='cuda'):
            with torch.no_grad():
                start_time = time.time()
                model_logits = self.net(input_tensor)
                end_time = time.time()

                mask = torch.sigmoid(model_logits) if threshold is None else torch.sigmoid(model_logits) > threshold
                mask = mask.squeeze(0).detach().cpu().numpy()

        inference_time = end_time - start_time
        log.info(f'[PYTORCH]: Inference prediction took {(inference_time * 1000):.2f} ms.')
        return (mask[0], inference_time)