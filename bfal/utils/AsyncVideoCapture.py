import cv2 as cv
from threading import Thread
import time

class AsyncVideoCapture:

    def __init__(self, target=0) -> None:

        self.cap = cv.VideoCapture(target)
        self.ret, self.frame = False, None

    def __start_capture__(self) -> None:
        while self.cap.isOpened():
            self.ret, self.frame = self.cap.read()
            time.sleep(0.05) # wait for 50000 nanoseconds

        return self

    def read(self):
        return (self.ret, self.frame)

    def begin(self) -> None:
        self.ret, self.frame = self.cap.read()
        self.__thread__ = Thread(target=self.__start_capture__)
        self.__thread__.start()
    
    def release(self) -> None:
        self.cap.release()

    def set(self, propId: int, value: float) -> None:
        self.cap.set(propId=propId, value=value)

    def get(self, propId: int) -> None:
        self.cap.get(propId=propId)
