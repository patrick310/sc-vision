import cv2
import PIL.Image, PIL.ImageTk
import pyzbar.pyzbar as pyzbar
import numpy as np
import tkinter

import logging
import time

logging.basicConfig(level=logging.DEBUG)


logger = logging.getLogger()
logger.handlers = [] # This is the key thing for the question!

# Start defining and assigning your handlers here
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s: %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.deployment = Deployment(filters=[grayscale_filter, barcode_filter, barcode_filter, barcode_filter])

        self.photo = None
        self.canvas = tkinter.Canvas(window, width=self.deployment.video.width, height=self.deployment.video.height)
        self.canvas.pack()

        self.delay = 15
        self.update()

        self.window.mainloop()

    def __del__(self):
        if self.deployment.video.cap.isOpened():
            self.deployment.video.cap.release()
        self.window.mainloop()

    def update(self):

        frame = self.deployment.process_frame()
        self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

        self.window.after(self.delay, self.update)


def barcode_filter(image, visualize=True):

    def decode(im):
        # Find barcodes and QR codes
        decodedObjects = pyzbar.decode(im)

        # Print results
        for obj in decodedObjects:
            print('Type : ', obj.type)
            print('Data : ', obj.data, '\n')

        return decodedObjects

    # Display barcode and QR code location
    def display(im, decodedObjects):

        # Loop over all decoded objects
        for decodedObject in decodedObjects:
            points = decodedObject.polygon

            # If the points do not form a quad, find convex hull
            if len(points) > 4:
                hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
                hull = list(map(tuple, np.squeeze(hull)))
            else:
                hull = points;

            # Number of points in the convex hull
            n = len(hull)

            # Draw the convex hull
            for j in range(0, n):
                cv2.line(im, hull[j], hull[(j + 1) % n], (255, 0, 0), 3)

    decoded_objects = decode(image)
    if len(decoded_objects) is not 0:
        logger.info(decoded_objects)

    if visualize:
        display(image, decoded_objects)

    return image


def grayscale_filter(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return gray


class VideoCapture:

    def __init__(self, video_source="C:\\Users\\patri\\Videos\\test_video.mkv"):

        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            raise ValueError("Unable to open video source", video_source)

        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                return ret, None
        else:
            return False, None


class Deployment:
    '''All filters must take an image (write an assert)
    '''

    def __init__(self, filters=None, video=None):
        if filters is None:
            filters = []
        if video is None:
            video = VideoCapture()

        self.filters = filters
        self.video = video

    def get_frame(self):
        return self.video.get_frame()

    def process_frame(self):
        start_time = time.time()
        ret, frame = self.get_frame()

        for method in self.filters:
            frame = method(image=frame)

        logger.debug('Frame process time was {0:0.1f} seconds'.format(
            time.time() - start_time) + ' for an FPS of {0:0.1f}'.format(
            1 / (time.time() - start_time)))

        return frame


if __name__ == '__main__':
    App(tkinter.Tk(), "AutoLens: Intelligent Inspection Solutions")

