import cv2
import videoio.interface as vi
from processing import background as bg
from utilities import cmdline as u
from utilities import model_loader
from videoio import source, receiver


def make_source(args) -> vi.ImageSource:
    t = args.source
    if t == u.SOURCE_FILE:
        return source.ImageFile(args.input_image_file)
    if t == u.SOURCE_WEBCAM:
        return source.WebCam(filename=args.webcam)
    raise ValueError("Unknown source type \"{}\"".format(str(t)))


def make_receiver(args) -> vi.ImageReceiver:
    t = args.destination
    if t == u.DEST_FAKE_WEBCAM:
        return receiver.FakeCam(args.fakecam)
    if t == u.DEST_IMAGE_FILE:
        return receiver.ImageFile(args.output_image_file)
    if t == u.DEST_WINDOW:
        return receiver.CV2Window("Fuel background substitution")
    return None

if __name__ == "__main__" :
    args = u.get_args()
    src = make_source(args)
    out = make_receiver(args)
    model = model_loader.load_segmentation_model(None, "GPU")
    processor = bg.Background(model=model,
                              blur=args.mode == u.MODE_BLUR,
                              ksize=args.blur_kernel_size,
                              bg_color=(120, 120, 120),
                              bg_filename=args.background
                              )

    try:
        while (cv2.waitKey(1) != 27):
            img = src.get_image()
            res = processor.process(img)
            out.receive_image(res)
    except KeyboardInterrupt:
        print("Stop image processing.")
