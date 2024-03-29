from typing import Tuple

import cv2
import numpy as np
import openvino as ov


class Background:
    _model:ov.CompiledModel = None
    _bg_color = (70, 70, 70)
    _bg_image:cv2.Mat = None
    _bg_filename:str = None
    _blur:bool = False
    _ksize:int = None

    def __init__(self, model:ov.CompiledModel,
                 bg_color:Tuple[int, int, int] = None,
                 bg_filename:str=None,
                 blur:bool = False,
                 ksize:int=None):
        self._model = model
        self._blur = blur
        if ksize is not None:
            self._ksize = ksize

        if bg_color is not None:
            self._bg_color = bg_color

        if bg_filename is not None:
            self._bg_filename = bg_filename

    def process(self, img:cv2.Mat) -> cv2.Mat:
        res = self._process_img(img)
        #res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
        return res

    def _process_img(self, img:cv2.Mat):
        img = self._equalize_hist(img)
        original_img = img.copy()
        img = self._adjust_gamma(img, 2.5)
        padded_img, pad_info = self._resize_and_pad(np.array(img))
        normalized_img = np.expand_dims(padded_img.astype(np.float32) / 255, 0)
        out = self._model(normalized_img)[0]
        image_data = np.array(original_img)
        orig_img_shape = image_data.shape
        mask = self._postprocess_mask(out, pad_info, orig_img_shape[:2])
        bg_image = self.get_bg_image(original_img)
        condition = np.stack((mask,) * 3, axis=-1) > 0.5
        output_image = np.where(condition, image_data, bg_image)
        #output_image = cv2.edgePreservingFilter(output_image, flags=1, sigma_s=64, sigma_r=0.2)
        #output_image = cv2.detailEnhance(output_image, sigma_s=10, sigma_r=0.15)
        return output_image.astype(np.uint8)

    def _adjust_gamma(self, img:cv2.Mat, gamma=1.0) ->cv2.Mat :
        gamma = 1.0 / gamma
        tbl = np.array([((i / 255.0) ** gamma) * 255
                        for i in np.arange(0, 256)]).astype(np.uint8)
        return cv2.LUT(img, tbl)

    def _equalize_hist(self, img:cv2.Mat) -> cv2.Mat :
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(2, 2))
        yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)

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
        mask = np.argmax(out, -1)[0]
        pad_h, pad_w = pad_info
        unpad_h = mask.shape[0] - pad_h
        unpad_w = mask.shape[1] - pad_w
        mask_unpadded = mask[:unpad_h, :unpad_w]
        orig_h, orig_w = orig_img_size
        mask = cv2.resize(mask_unpadded, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        return mask

    def get_bg_image(self, img:cv2.Mat):
        bg_image = None
        if self._blur:
            if self._ksize is None:
                self._ksize = 55
            if self._ksize < 5:
                self._ksize = 5
            if self._ksize > 105:
                self._ksize = 105
            bg_image = cv2.GaussianBlur(img, (self._ksize, self._ksize), 0, cv2.BORDER_DEFAULT)
        else:
            shape = np.array(img).shape
            if self._bg_filename is not None and len(self._bg_filename) > 0 :
                if self._bg_image is not None and np.array(self._bg_image).shape != shape:
                    self._bg_image = None
                if self._bg_image is None:
                    bg_image = cv2.imread(self._bg_filename, cv2.IMREAD_REDUCED_COLOR_4)
                    bg_image = cv2.resize(bg_image, dsize=(shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
                    bg_image = self._equalize_hist(bg_image)
                else:
                    bg_image = self._bg_image.copy()
            else:
                bg_image = np.full(shape, self._bg_color, dtype=np.uint8)
        return bg_image