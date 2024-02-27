import cv2
import pyfakewebcam as fcam

from videoio.interface import ImageReceiver


class CV2Window(ImageReceiver):
    _name:str = "Video output."
    def __init__(self, name:str=None):
        if name is not None and len(name) > 0:
            self._name = name

    def __del__(self):
        cv2.destroyAllWindows()

    def receive_image(self, image:cv2.Mat):
        if image is None:
            raise RuntimeError("Failed to output an empty image.")
        cv2.namedWindow(self._name, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(self._name, image)

class ImageFile(ImageReceiver):
    _filename = None
    def __init__(self, filename:str):
        if filename is None or len(filename) < 1:
            raise ValueError("You can't use an empty path to same the result image.")
        self._filename = filename

    def receive_image(self, image: cv2.Mat):
        cv2.imwrite(self._filename, image)


class FakeCam(ImageReceiver):
    _cam = None
    def __init__(self, device:str, width:int = 800, height:int = 600, fps:int = 30):
        self._cam = fcam.FakeWebcam(video_device=device,
                                            width=width, height=height,
                                            channels=3)

    def receive_image(self, image:cv2.Mat):
        self._cam.schedule_frame(image)
