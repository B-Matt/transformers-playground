import cv2
import time
import torch
import pathlib

import numpy as np
# import tensorrt as trt

# import pycuda.autoinit
# import pycuda.driver as cuda

from typing import Any, Tuple
from utils.logging import logging

# Logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# trt_logger = trt.Logger(trt.Logger.INFO)

# Classes
"""
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class TensorRTEngine:
    def __init__(self, engine_file_path: str, image_resolution: Tuple[int, int]):
        log.info("[ENGINE]: TensorRT engine inference object initialized.")
        self.engine_file_path: pathlib.Path = pathlib.Path(engine_file_path)
        if not self.engine_file_path.exists():
            raise ValueError("The engine file does not exist.")
        log.info(f"[TensorRT]: File exists at path: {engine_file_path}.")

        self.engine = TensorRTEngine.load_engine(engine_file_path=self.engine_file_path)
        self.context = self.engine.create_execution_context()
        log.info(f"[TensorRT]: Engine type: {type(self.engine)}.")

        self.image_resolution = image_resolution
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        log.info("[ENGINE]: TensorRT engine inference object initialized.")

    def inference_width(self) -> int:
        return self.image_resolution[1]

    def inference_height(self) -> int:
        return self.image_resolution[0]

    @classmethod
    def load_engine(cls, engine_file_path: pathlib.Path) -> Any:
        with open(engine_file_path, mode="rb") as reader, trt.Runtime(trt_logger) as trt_runtime:
            engine = trt_runtime.deserialize_cuda_engine(reader.read())
        return engine

    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        # stream = cuda.Stream()

        # Allocate host and device buffers
        for binding in self.engine:
            binding_idx = self.engine.get_binding_index(binding)
            size = trt.volume(self.context.get_binding_shape(binding_idx))

            host_memory = cuda.pagelocked_empty(size, np.float32)
            device_memory = cuda.mem_alloc(host_memory.nbytes)
            bindings.append(int(device_memory))

            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_memory, device_memory))
            else:
                outputs.append(HostDeviceMem(host_memory, device_memory))
        return inputs, outputs, bindings, stream

    def run_inference(
        self,
        input_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        'input_tensor' expected to contain 4 dimensional data - (N, C, H, W)
        input_array = input_tensor.numpy()
        np.copyto(self.inputs[0].host, input_array.ravel())

        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)

        start_time = time.time()
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        end_time = time.time()

        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)

        self.stream.synchronize()

        output_as_numpy_matrix = np.reshape(out.host, (self.inference_height(), self.inference_width()))
        return (torch.Tensor(output_as_numpy_matrix), (end_time - start_time) * 1000)

    def predict_image(
        self,
        input_tensor: np.ndarray,
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, int]:

        # Run Inference
        model_logits, inference_time = self.run_inference(input_tensor)

        mask = torch.sigmoid(model_logits) if threshold is None else torch.sigmoid(model_logits) > threshold
        mask = mask.squeeze(0).detach().cpu().numpy()
        #model_output = torch.permute(model_output, (1, 2, 0)).numpy()

        log.info(f'[TensorRT]: Inference prediction took {(inference_time * 1000):.2f} ms.')     
        return (mask[0], inference_time)
"""