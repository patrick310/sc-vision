import CameraEngine
import TimeEngine
import MessageEngine
import JSONEngine
import RPi.GPIO as GPIO
import cv2
import threading
from keras import *
from keras.models import *
import numpy as np
import os, sys

config_loader = JSONEngine.JSONEngine(filepath = "config.json")
configs = config_loader.load()

Cam = CameraEngine.CameraEngine(
        resolution = (configs["camera"]["resolution"][0], configs["camera"]["resolution"][1]),
        exposure_mode = configs["camera"]["exposure_mode"],
        status_LED_enabled = configs["camera"]["status_LED_enabled"],
        status_LED_pin = configs["camera"]["status_LED_pin"],
        ring_LED_enabled = configs["camera"]["ring_LED_enabled"],
        ring_LED_pin = configs["camera"]["ring_LED_pin"],
        error_LED_enabled = configs["camera"]["error_LED_enabled"],
        error_LED_pin = configs["camera"]["error_LED_pin"],
        discard_blurry_enabled = configs["camera"]["discard_blurry_enabled"],
        blurry_threshold = configs["camera"]["blurry_threshold"],
        photo_limit_enabled = configs["camera"]["photo_limit_enabled"],
        photo_limit = configs["camera"]["photo_limit"]
        )

Time = TimeEngine.TimeEngine(
        time_limit_enabled = configs["time"]["time_limit_enabled"],
        hours_to_run = configs["time"]["hours"],
        minutes_to_run = configs["time"]["minutes"],
        seconds_to_run = configs["time"]["seconds"],
        capture_interval = configs["time"]["capture_interval"]
        )

Mess = MessageEngine.MessageEngine(
        logfile_enabled = configs["messages"]["logfile_enabled"],
        logfile_path = configs["messages"]["logfile_path"]
        )

model = load_model(configs["models"]["neural_net_filepath"])

Mess.handle_message("All configurations read.", MessageEngine.INFO_IDENT)

threads = list()
plock = threading.Lock()

def pretty_date(capture_time):
    result = list()
    result.append(str(capture_time.hour))
    result.append(":")
    result.append(str(capture_time.minute))
    result.append(":")
    result.append(str(capture_time.second))
    result.append("-")
    result.append(str(capture_time.day))
    result.append("-")
    result.append(str(capture_time.month))
    result.append("-")
    result.append(str(capture_time.year))
    return "".join(result)

def process_image(image, image_number, capture_time):
    global model
    global plock
    global configs
    mod_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mod_image = cv2.resize(mod_image, (150, 150))
    temp = list()
    temp.append(mod_image)
    mod_image = np.array(temp)
    mod_image = mod_image.reshape(mod_image.shape[0], 150, 150, 1)
    mod_image = mod_image.astype('float32')
    mod_image /= 255
    result = model.predict_proba(mod_image, batch_size = 1, verbose = 0)[0]
    highest_proba_index = 0
    for index in range(len(result)):
        if result[index] > result[highest_proba_index]:
            highest_proba_index = index
    path = None
    class_names = ["asd", "solid", "standard", "background"]
    for cn in class_names:
        if highest_proba_index == configs["models"]["classes"][cn]["index"]:
            path = configs["models"]["classes"][cn]["path"]
            break

    try:
        plock.acquire()
        msg = "Saving " + configs["image_descriptor"] + str(image_number) + "_" + pretty_date(capture_time) + " to " + path
        Mess.handle_message(msg, MessageEngine.INFO_IDENT)
    	CWD = os.getcwd()
    	os.chdir(CWD + path)
    	name = configs["image_descriptor"] + str(image_number) + "_" + pretty_date(capture_time) + ".jpg"
    	cv2.imwrite(name, image)
    	os.chdir(CWD)
        plock.release()
    except Exception as err:
        msg = "couldn't save image with error message: " + str(err)
        Mess.handle_message(msg, MessageEngine.ERROR_IDENT)
        plock.release()

try:
    while not Cam.photo_limit_reached() and not Time.time_limit_reached():
        if threading.active_count() >= 10:
            Mess.handle_message('System bogged. Joining threads', MessageEngine.INFO_IDENT)
            for thread in threads:
                thread.join()
            for _ in range(len(threads)):
                threads.pop()
            print "tc: " + str(len(threads))
        Time.start_interval_timer()
        image, image_number, capture_time  = Cam.capture()
        process_thread = threading.Thread(target = process_image, args = ([image, image_number, capture_time]))
        process_thread.start()
        threads.append(process_thread)
        Time.wait_for_interval_expiration()
    for thread in threads:
        thread.join()
    Mess.handle_message("done", MessageEngine.INFO_IDENT)
    GPIO.cleanup()
except Exception as err:
    GPIO.cleanup()
    Mess.handle_message("done", MessageEngine.ERROR_IDENT)
    raise err

for thread in threads:
	thread.join()

# EOF
