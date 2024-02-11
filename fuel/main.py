import openvino as ov
from pathlib import Path
import cv2
import numpy as np
from typing import Tuple, Union
import pyvirtualcam


core = ov.Core()

tflite_model_path = Path("../models/segmentation/selfie_multiclass_256x256")
ir_model_path = tflite_model_path.with_suffix(".xml")

ov_model = core.read_model(ir_model_path)
compiled_model = core.compile_model(ov_model, "GPU")



# Preprocessing helper function
def resize_and_pad(image:np.ndarray, height:int = 256, width:int = 256):
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
def postprocess_mask(out:np.ndarray, pad_info:Tuple[int, int], orig_img_size:Tuple[int, int]):
    label_mask = np.argmax(out, -1)[0]
    pad_h, pad_w = pad_info
    unpad_h = label_mask.shape[0] - pad_h
    unpad_w = label_mask.shape[1] - pad_w
    label_mask_unpadded = label_mask[:unpad_h, :unpad_w]
    orig_h, orig_w = orig_img_size
    label_mask_resized = cv2.resize(label_mask_unpadded, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return label_mask_resized

def process_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    padded_img, pad_info = resize_and_pad(np.array(img))
    normalized_img = np.expand_dims(padded_img.astype(np.float32) / 255, 0)

    out = compiled_model(normalized_img)[0]
    image_data = np.array(img)
    orig_img_shape = image_data.shape
    postprocessed_mask = postprocess_mask(out, pad_info, orig_img_shape[:2])
    BG_COLOR = (70, 70, 70)
    bg_image = np.full(orig_img_shape, BG_COLOR, dtype=np.uint8)
    condition = np.stack((postprocessed_mask,) * 3, axis=-1) > 0.85
    output_image = np.where(condition, image_data, bg_image)
    return output_image.astype(np.uint8)

def run_background_blurring(source:Union[str, int] = 0, flip:bool = True, use_popup:bool = True, skip_first_frames:int = 0, model:ov.Model = ov_model, device:str = "GPU"):
    cap = None
    cam = None
    try:
        cap = cv2.VideoCapture(source)
        #cam = pyvirtualcam.Camera(device="/dev/video3", width=640, height=480, fps=30)
        if not cap.isOpened():
            print("Failed to open webcam")
            exit(0)
        cv2.namedWindow(
            winname="Img", flags=cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE
        )

        while True:
            _, frame = cap.read()
            frame = cv2.flip(frame, 1)
            res = process_img(frame)
            res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
            cv2.imshow("Img", res)
            #cam.send(res)

            if cv2.waitKey(1) == 27:
                break
    except RuntimeError as e:
        print(e)
    finally:
        if cap is not None:
            cap.release()
            cv2.destroyAllWindows()




VIDEO_SOURCE = 0

run_background_blurring(source=VIDEO_SOURCE, device="GPU")
