import cv2

class ImageSource:
    def get_image(self) -> cv2.Mat:
        pass

class ImageReceiver:
    def receive_image(self, image:cv2.Mat):
        pass