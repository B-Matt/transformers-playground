import time
import torch
import pathlib
import onnxruntime

import numpy as np

from typing import Tuple
from utils.logging import logging

# Logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Engine Class
class OnnxEngine:
    def __init__(
        self,
        onnx_model_path: str,
        image_resolution: Tuple[int, int]
    ):
        log.info(f'[ENGINE]: Started ONNX engine loading!')
        self.onnx_model_path: pathlib.Path = pathlib.Path(onnx_model_path)        
        self.image_resolution = image_resolution
        log.info("[ENGINE]: Onnx engine inference object initialized.")

    def warmup(self, warmup_iters: int) -> None:
        log.info(f'[ONNX]: Model warm up for {warmup_iters} iteration/s.')
        dummy_input = torch.rand((1, 3, self.patch_w, self.patch_h))

        for i in range(1, warmup_iters):
            self.predict_image(dummy_input, None, False)

    def predict_image(
        self,
        input_tensor: torch.Tensor,
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, int]:
        """
        'input_tensor' expected to contain 4 dimensional data - (N, C, H, W)
        """
        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = [
            ('CUDAExecutionProvider', {
                'cudnn_conv_use_max_workspace': '1',
                'cudnn_conv_algo_search': 'EXHAUSTIVE'
            }),
            'CPUExecutionProvider',
        ]
        onnx_session = onnxruntime.InferenceSession(self.onnx_model_path, options, providers)

        input_name = onnx_session.get_inputs()[0].name
        output_name = onnx_session.get_outputs()[0].name
        io_binding = onnx_session.io_binding()

        input_tensor = input_tensor.contiguous()

        io_binding.bind_input(
            name=input_name,
            device_type='cuda',
            device_id=0,
            element_type=np.float32,
            shape=tuple(input_tensor.shape),
            buffer_ptr=input_tensor.data_ptr(),
        )
        io_binding.bind_output(output_name, 'cuda')

        start_time = time.time()
        onnx_session.run_with_iobinding(io_binding)      # run([output_name], { input_name: input_numpy_array })
        end_time = time.time()

        model_output = io_binding.copy_outputs_to_cpu()

        inference_time = end_time - start_time
        log.info(f'[ONNX]: Inference prediction took {(inference_time * 1000):.2f} ms.')
        return (model_output[0], inference_time)
