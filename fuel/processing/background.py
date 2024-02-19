import cv2
import openvino as ov
import numpy as np
from typing import Tuple, Union

class Background:
    _model:ov.CompiledModel = None
    _bg_color = (70, 70, 70)

    def __init__(self, model:ov.CompiledModel,
                 bg_color:Tuple[int, int, int] = None):
        self._model = model
        if bg_color is not None:
            self._bg_color = bg_color

    def process(self, img:cv2.Mat) -> cv2.Mat:
        res = self._process_img(img)
        res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
        return res

    def _process_img(self, img:cv2.Mat):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        padded_img, pad_info = self._resize_and_pad(np.array(img))
        normalized_img = np.expand_dims(padded_img.astype(np.float32) / 255, 0)
        out = self._model(normalized_img)
        out = out[0]
        image_data = np.array(img)
        orig_img_shape = image_data.shape
        postprocessed_mask = self._postprocess_mask(out, pad_info, orig_img_shape[:2])
        bg_image = np.full(orig_img_shape, self._bg_color, dtype=np.uint8)
        condition = np.stack((postprocessed_mask,) * 3, axis=-1) > 0.85
        output_image = np.where(condition, image_data, bg_image)
        return output_image.astype(np.uint8)

    def _resize_and_pad(self, image: np.ndarray, height: int = 256, width: int = 256):
        h, w = image.shape[:2]
        if h < w:
            img = cv2.resize(image, (width, np.floor(h / (w / width)).astype(int)))
        else:
            img = cv2.resize(image, (np.floor(w / (h / height)).astype(int), height))

        r_h, r_w = img.shape[:2]
        right_padding = width - r_w
        bottom_padding = height - r_h
        padded_img = cv2.copyMakeBorder(img, 0, bottom_padding, 0, right_padding, cv2.BORDER_CONSTANT)
        return padded_img, (bottom_padding, right_padding)

    def _postprocess_mask(self, out: np.ndarray, pad_info: Tuple[int, int], orig_img_size: Tuple[int, int]):
        label_mask = np.argmax(out, -1)[0]
        pad_h, pad_w = pad_info
        unpad_h = label_mask.shape[0] - pad_h
        unpad_w = label_mask.shape[1] - pad_w
        label_mask_unpadded = label_mask[:unpad_h, :unpad_w]
        orig_h, orig_w = orig_img_size
        label_mask_resized = cv2.resize(label_mask_unpadded, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        return label_mask_resized