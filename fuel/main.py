import cv2

from utilities import cmdline as u
from utilities import model_loader
from processing import background as bg
import videoio.interface as vi
from videoio import source, receiver

def make_source(args) -> vi.ImageSource:
    t = args.get(u.ARG_SOURCE)
    if t == u.SOURCE_FILE:
        return source.ImageFile(args.get(u.ARG_SRC_IMAGE_FILE))
    if t == u.SOURCE_WEBCAM:
        return source.WebCam(args.get(u.ARG_SRC_WEBCAM))
    raise ValueError("Unknown source type \"{}\"".format(str(t)))

def make_receiver(args) -> vi.ImageReceiver:
    t = args.get(u.ARG_DESTINATION)
    if t == u.DEST_FAKE_WEBCAM:
        return receiver.FakeCam(args.get(u.ARG_DEST_FAKECAM))
    if t == u.DEST_IMAGE_FILE:
        return receiver.ImageFile(args.get(u.ARG_DEST_IMAGE_FILE))
    if t == u.DEST_WINDOW:
        return receiver.CV2Windown("Fuel background substitution")
    return None

if __name__ == "__main__" :
    args = u.get_args()
    src = make_source(args)
    out = make_receiver(args)
    model = model_loader.load_segmentation_model(None, "GPU")
    processor = bg.Background(model, bg_color=(120, 120, 120))

    while (True):
        if cv2.waitKey(10) == 27:
            break
        img = src.get_image()
        res = processor.process(img)
        out.receive_image(res)
