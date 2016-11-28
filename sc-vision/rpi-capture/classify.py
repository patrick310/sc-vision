import picamera
import numpy as np
from PIL import Image, ImageOps
import time, datetime
import io
from keras.models import load_model
import threading
import json
import configs
import os
import RPi.GPIO as GPIO
import logging
import mopiapi

# Takes an image packet and spawns a thread to process the image packet.
def process_image(image_packet):
    # _process_image runs in a different thread than the main thread, allowing
    # for concurrent operations to take place.  This allows for the picamera to
    # keep capturing images while images are being processed.
    def _process_image(image_packet):
        logging.debug("Starting image processing (" + str(threading.current_thread()) + ").")
        image = image_packet[0]
        image_number = image_packet[1]
        global size
        # Formats a PIL image into a numpy array that can be directly fed into
        # a neural network model.
        def format_image_for_network(image):
            arr = np.array(image)
            arr = arr.reshape(1, size[0], size[1])
            arr = arr.astype('float32')
            arr /= 255.0
            temp = list()
            temp.append(arr)
            arr = np.array(temp)
            return arr
        # Determines the highest value in an array and returns the index of that
        # value.  This is used to determine which directory to save a
        # captured image to.
        def highest_index(arr):
            highest_index = 0
            for index in range(len(arr)):
                if arr[index] > arr[highest_index]:
                    highest_index = index
            return highest_index
        # Takes a numpy array representing a single image and returns an array
        # containing class probabilities.
        def get_image_prediction(arr):
            global model
            return model.predict_proba(arr, batch_size = 1, verbose = 0)[0]
        # Scales a full-size image down to the appropriate size and converts the
        # scaled down image to grayscale.  The resulting resized, grayscale
        # image is returned (a copy--the original image is not affected).
        def reformat_image(image):
            return ImageOps.grayscale(image.resize(size))
        # Saves an image to the directory of the class it is most likely to be.
        def save_image(index):
            # Gets the directory name of the class the image is most likely to be.
            def get_dir_name():
                names = sorted(os.listdir(os.getcwd() + configs.save_folder))
                if len(names) != configs.nb_classes:
                    logging.critical("Wrong number of class directories in save directory (" + str(len(names)) + " =/= " + str(configs.nb_classes) + ")")
                    raise ValueError("Wrong number of class directories in save directory (" + str(len(names)) + " =/= " + str(configs.nb_classes) + ")")
                return names[index]

            try:
                image_name = os.getcwd() + configs.save_folder + "/" + get_dir_name() + "/" + configs.image_descriptor + str(image_number) + ".jpeg"
                image.save(image_name)
                logging.info("Saved image: " + image_name)
                logging.debug("Image processsed and saved (" + str(threading.current_thread()) + ").")
            except Exception as err:
                logging.error("Image failed to be processsed (" + str(threading.current_thread()) + ").")
                raise err

        # Copy the image to work with so that the full-size image can be
        # saved to the appropriate directory.
        image_copy = reformat_image(image)
        arr = format_image_for_network(image_copy)
        prediction = get_image_prediction(arr)
        index = highest_index(prediction)
        save_image(index)
    # Process the image_packet in another thread besides the main thread.
    thread = threading.Thread(target = _process_image, args = ([image_packet]))
    thread.start()
    # Add the thread to the global thread list to make sure the thread
    # completes its operation before the main thread stops execution.
    global threads
    threads.append(thread)

# Determines whether the time limit is reached as specified by the
# time limit values in the configs.json file.
def time_limit_reached(program_start_time):
    # Returns the current time.
    def now():
        return datetime.datetime.fromtimestamp(time.time())
    # Returns the total seconds elapsed between two times.
    def difference_between_times(start_time, stop_time = None):
        if stop_time is None:
            stop_time = now()
        time_difference = stop_time - start_time
        seconds_elapsed = time_difference.total_seconds()
        return seconds_elapsed
    # Calculates the total amount of time the program is supposed to run.
    def calculate_runtime():
        runtime = 0.0
        runtime += float(configs.time_limit_hours) * 3600.0
        runtime += float(configs.time_limit_minutes) * 60.0
        runtime += float(configs.time_limit_seoncds)
        return runtime

    if configs.time_limit_enabled:
        seconds_elapsed = difference_between_times(program_start_time)
        return seconds_elapsed >= calculate_runtime()
    else:
        return False

# Determines if the image capture limit is reached as specified by the
# configs.json file.
def image_limit_reached(current_count):
    if configs.image_limit_enabled:
        return current_count >= configs.image_limit_count
    else:
        return False

# Captures an image and returns a PIL image object.
def capture_image(
        resolution = (3280, 2464),
        exposure_mode = 'auto',
        warmup_delay = 0.2,
        format = 'jpeg'
        ):
    logging.debug("Capturing image.")
    GPIO.output(configs.camera_pin, GPIO.HIGH)
    stream = io.BytesIO()
    with picamera.PiCamera() as cam:
        cam.resolution = resolution
        cam.exposure_mode = exposure_mode
        time.sleep(warmup_delay)
        cam.capture(stream, format = format)
        cam.close()
    stream.seek(0)
    image = Image.open(stream)
    GPIO.output(configs.camera_pin, GPIO.LOW)
    logging.debug("Image captured.")
    return image

# Joins all threads to ensure that all captured images are processed and
# turns off all GPIO pins.
def cleanup():
    logging.info("Cleaning up.")
    def join_threads():
        global threads
        for thread in threads:
            thread.join()
    join_threads()
    GPIO.cleanup()
    logging.info("All clean.")

# The main program loop.
def loop():
    # Determines if the capture delay-- the time in between image captures--
    # has been reached.
    def capture_delay_reached(timer_start):
        return (datetime.datetime.fromtimestamp(time.time()) - timer_start).total_seconds() >= configs.capture_delay
    # A self-diagnostic function.  Uses the MoPi API to check the battery level
    # and checks to see if memory is getting low on the SD card.  If a critical
    # condition is met such as the battery being almost dead or the system is
    # out of memory, it will return True; otherwise False.
    def examine_conditions():
        pass

    global IMAGE_COUNT
    global PROGRAM_START_TIME
    logging.info("Starting main loop.")
    # Loop until a stop condition is met or an error/exception occurs.
    try:
        while not image_limit_reached(IMAGE_COUNT) and not time_limit_reached(PROGRAM_START_TIME):
            timer_start = datetime.datetime.fromtimestamp(time.time())
            image_packet = (capture_image(), IMAGE_COUNT)
            IMAGE_COUNT += 1
            process_image(image_packet)
            if examine_conditions():
                logging.critical("Critical condition reached.")
                cleanup()
                return
            while not capture_delay_reached(timer_start):
                pass
        logging.info("Exiting main loop b/c exit condition reached.")
        cleanup()
    except Exception as err:
        cleanup()
        logging.error("Exiting main loop b/c error occured.")
        print err

def setup():
    # Enable the use of the logging module.
    logging.basicConfig(
        filename = 'mylog.log',
        format = '%(asctime)s : %(levelname)s : %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S%p',
        level = logging.DEBUG)

    logging.info(" --- Beginning Program ---")

    logging.debug("Setting up global variables.")
    try:
        # Sets up the GPIO pins according the pin specificatins in the
        # configs.json file.
        def setup_GPIO():
            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BCM)

            GPIO.setup(configs.power_pin, GPIO.OUT)
            GPIO.setup(configs.camera_pin, GPIO.OUT)
            GPIO.setup(configs.error_pin, GPIO.OUT)
            GPIO.setup(configs.battery_pin, GPIO.OUT)
            GPIO.setup(configs.buzzer_pin, GPIO.OUT)
            GPIO.setup(configs.light_ring_pin, GPIO.OUT)
        setup_GPIO()

        global IMAGE_COUNT
        IMAGE_COUNT = 0
        global threads
        threads = list()
        global model
        logging.debug("Loading model.")
        model = load_model(configs.model_filepath)
        logging.debug("Model loaded.")
        global size
        size = (configs.img_width, configs.img_height)
        global PROGRAM_START_TIME
        PROGRAM_START_TIME = datetime.datetime.fromtimestamp(time.time())
        logging.debug("Fininshed setting up global variables.")
    except Exception as err:
        logging.error("Failed to set up all global variables.")
        raise err

setup()
loop()
