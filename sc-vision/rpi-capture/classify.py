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

save_lock = threading.Lock()

def process_image(image):
    global size
    def _process_image(image, size):
        def format_image_for_network(image):
            arr = np.array(image)
            arr = arr.reshape(1, size[0], size[1])
            arr = arr.astype('float32')
            arr /= 255.0
            temp = list()
            temp.append(arr)
            arr = np.array(temp)
            return arr
        def highest_index(arr):
            highest_index = 0
            for index in range(len(arr)):
                if arr[index] > arr[highest_index]:
                    print str(highest_index) + " -> " + str(index)
                    highest_index = index
            return highest_index
        def get_image_prediction(arr):
            global model
            return model.predict_proba(arr, batch_size = 1, verbose = 0)[0]
        def reformat_image(image, size):
            return ImageOps.grayscale(image.resize(size))
        def save_image(index):
            def get_dir_name():
                names = sorted(os.listdir(os.getcwd() + configs.save_folder))
                return names[index]
            global save_lock
            try:
                save_lock.acquire()
                global PHOTO_COUNT
                image.save(os.getcwd() + configs.save_folder + "/" + get_dir_name() + "/" + configs.image_descriptor + str(PHOTO_COUNT) + ".jpeg")
                #PHOTO_COUNT += 1
                save_lock.release()
            except Exception as err:
                save_lock.release()
                raise err

        image_copy = reformat_image(image, size)
        arr = format_image_for_network(image_copy)
        prediction = get_image_prediction(arr)
        index = highest_index(prediction)
        save_image(index)
        print "saved"
    thread = threading.Thread(target = _process_image, args = ([image, size]))
    thread.start()
    global threads
    threads.append(thread)


def setup_GPIO():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)

    GPIO.setup(configs.power_pin, GPIO.OUT)
    GPIO.setup(configs.camera_pin, GPIO.OUT)
    GPIO.setup(configs.error_pin, GPIO.OUT)
    GPIO.setup(configs.battery_pin, GPIO.OUT)
    GPIO.setup(configs.buzzer_pin, GPIO.OUT)
    GPIO.setup(configs.light_ring_pin, GPIO.OUT)

def time_limit_reached(program_start_time):
    def now():
        return datetime.datetime.fromtimestamp(time.time())
    def difference_between_times(start_time, stop_time = None):
        if stop_time is None:
            stop_time = now()
        time_difference = stop_time - start_time
        seconds_elapsed = time_difference.total_seconds()
        return seconds_elapsed
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

def photo_limit_reached(current_count):
    if configs.photo_limit_enabled:
        return current_count >= configs.photo_limit_count
    else:
        return False

def capture_image(
        resolution = (3280, 2464),
        exposure_mode = 'auto',
        warmup_delay = 0.2,
        format = 'jpeg'
        ):
    stream = io.BytesIO()
    with picamera.PiCamera() as cam:
        cam.resolution = resolution
        cam.exposure_mode = exposure_mode
        time.sleep(warmup_delay)
        cam.capture(stream, format = format)
        cam.close()
    stream.seek(0)
    image = Image.open(stream)
    return image

def cleanup():
    def join_threads():
        global threads
        for thread in threads:
            thread.join()
    join_threads()
    GPIO.cleanup()

def loop():
    try:
        while not photo_limit_reached(PHOTO_COUNT) and not time_limit_reached(PROGRAM_START_TIME):
            start = datetime.datetime.fromtimestamp(time.time())
            image = capture_image()
            PHOTO_COUNT += 1
            process_image(image)
            diff = datetime.datetime.fromtimestamp(time.time()) - start
            print diff.total_seconds(), configs.capture_delay
            while diff.total_seconds() < configs.capture_delay:
                diff = datetime.datetime.fromtimestamp(time.time()) - start
        cleanup()
    except Exception as err:
        cleanup()
        print err

logging.basicConfig(
    filename = 'mylog.log',
    format = '%(asctime)s : %(levelname)s : %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S%p',
    level = logging.DEBUG)
logging.info('starting')
get_power_source_update()
setup_GPIO()
PHOTO_COUNT = 0
threads = list()
model = load_model(configs.model_filepath)
size = (configs.img_width, configs.img_height)
PROGRAM_START_TIME = datetime.datetime.fromtimestamp(time.time())
loop()
