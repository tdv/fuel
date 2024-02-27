import os
import sys
from pathlib import Path

import openvino as ov

MAIN_SCRIPT_DIR = os.path.dirname(os.path.abspath(str(sys.modules['__main__'].__file__)))
DEFAULT_MODELS_DIR = "models"
SEGMENTATION_MODEL_DIR = "segmentation/selfie"
SEGMENTATION_MODEL_NAME = "selfie_multiclass_256x256"
TTF_SELFIE_MODEL_NAME = "selfie_multiclass_256x256.tflite"
TTF_SELFIE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
IR_MODEL_SUFFIX = ".xml"

def make_model_dir(path:Path):
    exists_path = path.exists()
    if exists_path and not path.is_dir():
        raise RuntimeError("There is a path \"" + path.name + "\" and the one is not a dir.")
    if not exists_path:
            os.makedirs(path)

def download_model(path:Path):
    make_model_dir(path)
    ttf_model_path = Path().joinpath(path, TTF_SELFIE_MODEL_NAME)
    exists_ttf = ttf_model_path.exists()
    if not exists_ttf:
        import urllib.request
        urllib.request.urlretrieve(TTF_SELFIE_MODEL_URL, ttf_model_path)

def convert_model(path:Path):
    ttf_model_path = Path().joinpath(path, TTF_SELFIE_MODEL_NAME)
    ir_model_path = ttf_model_path.with_suffix(IR_MODEL_SUFFIX)
    exists_ir_model = ir_model_path.exists()
    if not exists_ir_model:
        ir_model = ov.convert_model(ttf_model_path)
        ov.save_model(ir_model, ir_model_path)

def load_segmentation_model(path, device) -> ov.CompiledModel:
    if path is None or len(path) < 1:
        model_path = Path().joinpath(MAIN_SCRIPT_DIR, DEFAULT_MODELS_DIR,
                                     SEGMENTATION_MODEL_DIR, SEGMENTATION_MODEL_NAME)
        download_model(model_path)
        convert_model(model_path)
    ir_model_path = Path().joinpath(model_path, TTF_SELFIE_MODEL_NAME).with_suffix(IR_MODEL_SUFFIX)
    core = ov.Core()
    model =  core.read_model(ir_model_path)
    return core.compile_model(model, device)
