import picamera
import RPi.GPIO as GPIO
import io
import numpy as np
import cv2
import time
import datetime

class CameraEngine:
    MAX_CAM_RESOLUTION = (3280, 2464)
    DEFAULT_EXPOSURE_MODE = 'auto'

    def __init__(self,
                 resolution = None,
                 exposure_mode = None,
                 discard_blurry_enabled = False,
                 blurry_threshold = None,
                 photo_limit_enabled = False,
                 photo_limit = None,
                 status_LED_enabled = False,
                 status_LED_pin = None,
                 error_LED_enabled = False,
                 error_LED_pin = None,
                 ring_LED_enabled = False,
                 ring_LED_pin = None,
                ):
        if resolution is None:
            self.resolution = self.MAX_CAM_RESOLUTION
        else:
            self.resolution = resolution
        if exposure_mode is None:
            self.exposure_mode = self.DEFAULT_EXPOSURE_MODE
        else:
            self.exposure_mode = exposure_mode
        self.discard_blurry_enabled = discard_blurry_enabled
        self.blurry_threshold = blurry_threshold
        self.photo_limit_enabled = photo_limit_enabled
        self.photo_limit = photo_limit
        self.status_LED_enabled = status_LED_enabled
        self.status_LED_pin = status_LED_pin
        self.error_LED_enabled = error_LED_enabled
        self.error_LED_pin = error_LED_pin
        self.ring_LED_enabled = ring_LED_enabled
        self.ring_LED_pin = ring_LED_pin

        self.images_captured = 0

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        if self.status_LED_enabled:
            GPIO.setup(self.status_LED_pin, GPIO.OUT)
            GPIO.output(self.status_LED_pin, GPIO.LOW)
        if self.error_LED_enabled:
            GPIO.setup(self.error_LED_pin, GPIO.OUT)
            GPIO.output(self.error_LED_pin, GPIO.LOW)
        if self.ring_LED_enabled:
            GPIO.setup(self.ring_LED_pin, GPIO.OUT)
            GPIO.ouput(self.ring_LED_pin, GPIO.LOW)

    def _disable_status_LED(self):
        if self.status_LED_enabled:
            GPIO.output(self.status_LED_pin, GPIO.LOW)

    def _enable_status_LED(self):
        if self.status_LED_enabled:
            GPIO.output(self.status_LED_pin, GPIO.HIGH)

    def _disable_error_LED(self):
        if self.error_LED_enabled:
            GPIO.output(self.error_LED_pin, GPIO.LOW)

    def _enable_error_LED(self):
        if self.error_LED_enabled:
            GPIO.output(self.error_LED_pin, GPIO.HIGH)

    def _disable_ring_LED(self):
        if self.ring_LED_enabled:
            GPIO.output(self.ring_LED_pin, GPIO.LOW)

    def _enable_ring_LED(self):
        if self.ring_LED_enabled:
            GPIO.output(self.ring_LED_enabled, GPIO.HIGH)

    def _stream_to_cv2(self, stream):
        image_data = np.fromstring(stream.getvalue(), dtype = np.uint8)
        image = cv2.imdecode(image_data, 1)
        # This method yields a BGR image. If RGB image is desired,
        # uncommment the next line.
        # image = image[:, :, ::-1]
        return image

    def _sample_enviroment(self):
        if self.status_LED_enabled: self._enable_status_LED()
        try:
            cam = picamera.PiCamera()
            time.sleep(0.1)
        except:
            if self.error_LED_enabled: self._enable_error_LED()
            if self.status_LED_enabled: self._disable_status_LED()
            raise IOError("Could not connect to picamera.")

        try:
            cam.resolution = (200, 200)
            cam.exposure_mode = 'sports'
            stream = io.BytesIO()
            cam.capture(stream, 'jpeg')
            cam.close()
            image = self._stream_to_cv2(stream)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
            number_dark = 0
            for index in range(0, len(histogram) / 2):
                number_dark += histogram[index]
            number_light = 0
            for index in range(len(histogram) / 2, 256):
                number_light += histogram[index]
            if self.status_LED_enabled: self._disable_status_LED()
            if number_light > number_dark:
                return "light"
            else:
                return "dark"
        except Exception as err:
            cam.close()
            if self.status_LED_enabled: self._disable_status_LED()
            if self.error_LED_enabled: self._enable_error_LED()
            raise err

    def _image_is_blurry(self, image):
        if cv2.Laplacian(image, cv2.CV_64F).var() < self.blurry_threshold:
		print "Image is blurry!"
	return cv2.Laplacian(image, cv2.CV_64F).var() < self.blurry_threshold

    def photo_limit_reached(self):
        if self.photo_limit_enabled:
            return self.images_captured >= self.photo_limit
        else:
            return False

    def capture(self):
        current_enviroment = 'light'
        if self.ring_LED_enabled:
            current_enviroment = self._sample_enviroment()

        if self.status_LED_enabled: self._enable_status_LED()
        try:
            cam = picamera.PiCamera()
            time.sleep(0.1)
        except:
            if self.error_LED_enabled: self._enable_error_LED()
            if self.status_LED_enabled: self._disable_status_LED()
            raise IOError("Could not connect to picamera.")

        try:
            cam.resolution = self.resolution
            cam.exposure_mode = self.exposure_mode
            stream = io.BytesIO()

            if self.ring_LED_enabled and current_enviroment == 'dark':
                self._enable_ring_LED()
            cam.capture(stream, 'jpeg')
            if self.ring_LED_enabled and current_enviroment == 'dark':
                self._disable_ring_LED()
            image = self._stream_to_cv2(stream)

            while self.discard_blurry_enabled and self._image_is_blurry(image):
                del stream
                stream = io.BytesIO()
                if self.ring_LED_enabled and current_enviroment == 'dark':
                    self._enable_ring_LED()
                cam.capture(stream, 'jpeg')
                if self.ring_LED_enabled and current_enviroment == 'dark':
                    self._disable_ring_LED()
                image = self._stream_to_cv2(image)

            cam.close()
            if self.status_LED_enabled: self._disable_status_LED()
            self.images_captured += 1
            return image, self.images_captured, datetime.datetime.fromtimestamp(time.time())
        except Exception as err:
            cam.close()
            if self.status_LED_enabled: self._disable_status_LED()
            if self.error_LED_enabled: self._enable_error_LED()
            raise err
# EOF
