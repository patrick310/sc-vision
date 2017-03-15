import camera.py
from time import sleep

class CaptureProcess(camera, timedelay):

    def _init_(self,camera, delay ):
        self.camera = camera
        self.delay = timedelay
        self.loop_active = False


    def start_capture_loop(self):
        self.loop_active = True
        self.loop()
            

    def stop_capture_loop(self):
        self.loop_active = False

    def loop(self):
        while self.loop_active:
            camera.capture_to_file()
            sleep(delay)
            
if __name__ == "__main__":
    a = camera.TestCamera()
    b = CaptureProcess(a,2)
    b.start_capture_loop()
