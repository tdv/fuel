import cv2
from videoio.interface import ImageReceiver

class CV2Windown(ImageReceiver):
    _name:str = "Video output."
    def __init__(self, name:str=None):
        if name is not None and len(name) > 0:
            self._name = name

    def __del__(self):
        cv2.destroyAllWindows()

    def receive_image(self, image:cv2.Mat):
        if image is None:
            raise RuntimeError("Failed to output an empty image.")
        cv2.namedWindow(self._name, cv2.WINDOW_NORMAL)
        cv2.imshow(self._name, image)
