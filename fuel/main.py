import cv2
from utilities import cmdline, model_loader
from processing import background as bg
from videoio import source, receiver

flags = cmdline.get_flags()
model_path = flags.get("model_path")
model = model_loader.load_segmentation_model(model_path, "GPU")

# src = source.WebCam("/dev/video0")
src  = source.FileImage("/home/dmitry/Tst/test_ltorch_bg_remove/data/Ich_bin.jpg")
out = receiver.CV2Windown("Test output")

processor = bg.Background(model, bg_color=(120, 120, 120))

while (True):
    if cv2.waitKey(10) == 27:
        break
    img = src.get_image()
    res = processor.process(img)
    out.receive_image(res)

#import pyvirtualcam
#cam = pyvirtualcam.Camera(device="/dev/video3", width=640, height=480, fps=30)
