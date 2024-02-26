import cv2

from videoio.interface import ImageSource


class ImageFile(ImageSource):
    _img:cv2.Mat = None

    def __init__(self, filename:str):
        self._img = cv2.imread(filename)

    def get_image(self) -> cv2.Mat:
        return self._img

class WebCam(ImageSource):
    _flip:bool = True
    _cam:cv2.VideoCapture = None

    def __init__(self, filename:str, flip:bool=True):
        self._cam = cv2.VideoCapture(filename)
        if not self._cam.isOpened():
            raise RuntimeError("Failed to open webcam \"{}\"".format(filename))

        self._cam.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        self._cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

    def __del__(self):
        self._cam.release()

    def get_image(self) -> cv2.Mat:
        _, frame = self._cam.read()
        if self._flip:
            frame = cv2.flip(frame, 1)
        return frame
