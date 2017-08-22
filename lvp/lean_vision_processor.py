from keras.applications.resnet50 import ResNet50
from keras.backend import shape
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import keras

class LeanVisionProcessor():

    '''Class which represents a vision system instance. It contains the logic of the system, and methods to control
    device specific parameters such as logging, alarms, and storage. This lean version is designed to be run locally.'''

    def __init__(self):
        self.current_state = "Initialized"
        self.capturing = False
        self.model = None
        self.capture_mode = 'cv2'

        self.alarm_cases = []

    def start_capture(self):
        self.capturing = True

    def set_test_case(self):
        self.set_model(ResNet50(weights='imagenet'))

    def stop_capture(self):
        self.capturing = False

    def set_model(self, keras_model):
        self.model = keras_model

    def set_alarm_case(self, alarm_case):
        self.alarm_cases.append(alarm_case)

    def load_image(self, image_path):
        img = image.load_img(image_path, target_size=self.get_model_resolution())
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        return x

    def has_alarm(self, image_to_check):
        prediction = self.predict_top_classes(image_to_check, num_of_classes=1)[0]

        for alarm in self.alarm_cases:
            return alarm in prediction

    def get_model_resolution(self):
        s = self.model.inputs[0].get_shape()
        return tuple([s[i].value for i in range(0, len(s))])[1:3]

    def predict_top_classes(self, pp_image, num_of_classes=3):
        model = self.model

        predictions = model.predict(pp_image)
        return decode_predictions(predictions, top=num_of_classes)[0]

    def predict_top_class(self, pp_image):
        return self.predict_top_classes(pp_image, num_of_classes=1)[0]


if __name__ == '__main__':
    test = LeanVisionProcessor()
    test.set_model(ResNet50(weights='imagenet'))

