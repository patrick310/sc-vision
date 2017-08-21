from datetime import datetime
import configs
import cv2
from keras.models import load_model

from .helpers import format_image_for_network

class VisionDataEvaluator:

    def __init__(self):

        self.model_bg_vs_bolts = load_model("model_output_bg_vs_bolts.h5")
        self.model_two_vs_four = load_model("model_output_two_vs_four.h5")
        self.model_inner_vs_outer = load_model("model_output_inner_vs_outer.h5")

        F = open("log.txt", "w")
        F.write("[Info] " + datetime.now().strftime(configs.time_format) + " Start of session\n")

    def predict_image(self):
        global counter
        global bg_count
        global prediction
        global new_car

        prediction = self.model_bg_vs_bolts.predict(format_image_for_network(frame))
        class_labels = ['background', 'bolts [cascade terminated]']

        # For debugging with video and cv2.putText, you should get rid of the requirement of new_car == True
        # in the following if statement, as currently the cascading terminates once the image has been
        # determined to be bolts AND a car that has already been analyzed. The screen will show
        # "bolts [cascade terminated]" in such a case.

        # if picture is of bolts:
        if int(prediction[0][0]) == 1 and new_car == True:
            log_new_car()
            class_labels = ['4 bolt', 'You should never see this']
            prediction = self.model_two_vs_four.predict(format_image_for_network(frame))
            # if picture is of two bolts:
            if int(prediction[0][0]) == 1:
                class_labels = ['2 inner bolt', '2 outer bolt']
                prediction = self.model_inner_vs_outer.predict(format_image_for_network(frame))
                # if prediction is outer, record the image and log the car:
                if int(prediction[0][0]) == 1:
                    cv2.imwrite("automatic_data/two_outer/outer_" + file_name, oframe)
                    new_car = False
                    bg_count = 0
                    log_car_configuration(class_labels[1])
                    counter = counter + 1
                # else prediction is inner, record the image and log the car:
                elif int(prediction[0][0]) == 0:
                    cv2.imwrite("automatic_data/two_inner/inner_" + file_name, oframe)
                    new_car = False
                    bg_count = 0
                    log_car_configuration(class_labels[0])
                    counter = counter + 1
            # else picture is of four bolts, so record the image and log the car:
            elif int(prediction[0][0]) == 0:
                cv2.imwrite("automatic_data/four/four_" + file_name, oframe)
                new_car = False
                bg_count = 0
                log_car_configuration(class_labels[0])
                counter = counter + 1
        # else picture is of background, so add to the background count:
        elif int(prediction[0][0]) == 0:
            # Optional: must have [required_bg_frames] (currently 3) consecutive bg frames to consider a car "done."
            # We always expect several consecutive background frames between cars, and doing this
            # prevents a single incorrect evaluation of a background image from starting a "new car."
            # Otherwise, set configs.required_bg_frames to 1, as this will start a "new car" every background frame.
            # Not sure how common this might be, since it hasn't been tested.
            # The same thing could be implemented to avoid an incorrect bolt evaluation.
            bg_count = bg_count + 1
            if bg_count == configs.required_bg_frames:
                new_car = True
                if counter != 0:  # this ensures that if no car has been logged yet, we don't log a car as leaving
                    log_car_leaving()

        returned_class_id = class_labels[int(prediction[0][0])]

        return returned_class_id