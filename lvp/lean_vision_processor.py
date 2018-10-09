from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.resnet50 import preprocess_input, decode_predictions

import numpy as np
import logging
import cv2
import time
import os

from vdp.helpers import set_resolution


class LeanVisionProcessor:
    '''
    Class which represents a vision system instance. It contains the logic of the system, and methods to control
    device specific parameters such as logging, alarms, and storage. This lean version is designed to be run locally.
    '''

    def __init__(self):
        self.current_state = "Initialized"
        self.capturing = False
        self.model = None
        self.capture_mode = 'cv2'
        self.capture_device = 0
        self.preview = False

        self.alarm_cases = []
        self.save_cases = []

        self.classes = []
        self.set_classes()


        logging.info("LVP Initialized")

    def start_capture(self):
        self.capturing = True

        if self.capture_mode is 'cv2':
            if self.preview:
                cv2.namedWindow("preview")
            vc = cv2.VideoCapture(0)

            if vc.isOpened():  # try to get the first frame
                rval, frame = vc.read()
            else:
                rval = False

            set_resolution(vc, 1920, 1080)

            logging.info("Video capture started")

            while rval:
                rval, frame = vc.read()
                original_frame = frame.copy()

                if self.color_mode is 'grayscale':
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                self.frame_loop(frame)

                if self.preview:
                    cv2.imshow("preview", frame)

                key = cv2.waitKey(20)

                if key == 27:  # exit on ESC
                    self.stop_capture()

                if self.capturing is False:
                    break

    def set_test_case(self):
        self.set_model(ResNet50(weights='imagenet'))

    def stop_capture(self):
        self.capturing = False
        cv2.destroyWindow("preview")

    def set_model(self, keras_model):
        self.model = keras_model

    def set_classes(self, class_dictionary=None):
        self.classes = class_dictionary
        self.classes = {'vehicle_background': 4, 'between_cars': 3, '1bolt_inner': 0, '2bolt': 2, '1bolt_outer': 1}

        logging.info("Classes set to " + str(self.classes))

    def frame_loop(self, frame):
        prediction = self.predict_top_class(self.image_from_memory(frame))

        if self.has_alarm(prediction):
            logging.info("Alarm detected for: ", prediction[1])

        if self.has_save_flag(prediction):
            logging.info("File save flag detected for: ", prediction[1])
            if not os.path.isdir(str(prediction[1] + '/')):
                os.makedirs(str(prediction[1] + '/'))
                logging.info("No directory found, creating directory for ", prediction[1])
            file_name = str(prediction[1] + '/' + time.strftime("%Y%m%d-%H%M%S") + ".jpg")
            cv2.imwrite(file_name, frame)

    def set_alarm_case(self, alarm_case):
        self.alarm_cases.append(alarm_case)

    def set_file_save_case(self, file_save_case):
        self.save_cases.append(file_save_case)

    def image_from_file(self, image_path):
        img = image.load_img(image_path, target_size=self.get_model_resolution())
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        return x

    def image_from_memory(self, img):
        img = cv2.resize(img, self.get_model_resolution())
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        return x

    def has_alarm(self, prediction_to_check):
        prediction = prediction_to_check

        for alarm in self.alarm_cases:
            if alarm in prediction:
                return True
                break
            else:
                return False

    def has_save_flag(self, prediction_to_check):
        prediction = prediction_to_check

        for flag in self.save_cases:
            if flag in prediction:
                return True
                break
            else:
                return False

    def get_model_resolution(self):
        s = self.model.inputs[0].get_shape()
        return tuple([s[i].value for i in range(0, len(s))])[1:3]

    def get_model_color_mode(self):
        s = self.model.inputs[0].get_shape()
        s = tuple([s[i].value for i in range(0, len(s))])[3:4]
        s = s[0]

        if s is 3:
            self.color_mode = "rgb"

        elif s is 1:
            self.color_mode = "grayscale"

        else:
            raise ReferenceError

    def predict_top_classes(self, pp_image, num_of_classes=3):
        model = self.model

        predictions = model.predict(pp_image)
        return decode_predictions(predictions, top=num_of_classes)[0]

    def predict_top_class(self, pp_image):
        return self.predict_top_classes(pp_image, num_of_classes=1)[0]


if __name__ == '__main__':
    test = LeanVisionProcessor()
    test.set_test_case()
    print(test.get_model_color_mode())

