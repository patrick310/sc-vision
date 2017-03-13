import camera.py
from time import sleep

class CaptureProcess(camera, timedelay):

    def _init_(self,camera, delay ):
        self.camera = "----"
        self.delay = 500


    def start_capture_loop():
        #until stop_capture_loop is called
        #call capture
        sleep(0.2)          #delay 200 milli seconds

    def stop_capture_loop():
        
