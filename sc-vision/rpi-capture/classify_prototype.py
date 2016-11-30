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

class FLAG:
    def __init__(self,
            name = None,
            state = False):
        self.name = name
        self.state = state
        self.lock = threading.Lock()

    def name(self):
        return self.name

    def state(self):
        self.lock.acquire()
        _state = self.state
        self.lock.release()
        return _state

    def set_true(self):
        self.lock.acquire()
        self.state = True
        self.lock.release()
        return

    def set_false(self):
        self.lock.acquire()
        self.state = False
        self.lock.release()
        return

ERROR               = FLAG(name = "ERROR")
CRITICAL            = FLAG(name = "CRITICAL")
RUNNING             = FLAG(name = "RUNNING")
IDLING              = FLAG(name = "IDLING")
CAM_ON              = FLAG(name = "CAM_ON")
BATTERY_GOOD        = FLAG(name = "BATTERY_GOOD")
BATTERY_LOW         = FLAG(name = "BATTERY_LOW")
BATTERY_CRITICAL    = FLAG(name = "BATTERY_CRITICAL")
THREAD_STOP         = FLAG(name = "THREAD_STOP")

THREADS = list()

def now():
    return datetime.datetime.fromtimestamp(time.time())

def difference_between_times(start_time, stop_time = None):
    if stop_time is None:
        stop_time = now()
    time_difference = stop_time - start_time
    seconds_elapsed = time_difference.total_seconds()
    return seconds_elapsed

def thread_name():
    return str(threading.current_thread().name)

def blink(pin, delay, duration):
    timer_start = now()
    while difference_between_times(timer_start) < duration:
        GPIO.output(pin, GPIO.HIGH)
        time.sleep(delay)
        GPIO.output(pin, GPIO.LOW)

def solid(pin, duration):
    timer_start = now()
    GPIO.output(pin, GPIO.HIGH)
    while difference_between_times(timer_start) < duration:
        pass
    GPIO.output(pin, GPIO.LOW)

def update_error_LED():
    logging.debug("Starting 'update_error_LED()' in thread: " + thread_name())
    global THREAD_STOP
    global ERROR
    global CRITICAL
    # Loop while THREAD_STOP flag has not been raised.
    while not THREAD_STOP.state():
        # Critical error condition takes high priority.
        if CRITICAL.state():
            blink(pin = configs.error_pin, delay = 0.1, duration = 1.0)
        if ERROR.state():
            blink(pin = configs.error_pin, delay = 0.25, duration = 1.0)
        else:
            continue
    logging.debug("Exiting 'update_error_LED()' in thread: " + thread_name())

def update_power_LED():
    logging.debug("Starting 'update_power_LED()' in thread: " + thread_name())
    global THREAD_STOP
    global RUNNING
    global IDLING
    # Loop while THREAD_STOP flag has not been raised.
    while not THREAD_STOP.state():
        if IDLING.state():
            blink(pin = configs.power_pin, delay = 0.20, duration = 5.0)
        elif RUNNING.state():
            solid(pin = configs.power_pin, duration = 5.0)
        else:
            continue
    logging.debug("Exiting 'update_power_LED()' in thread: " + thread_name())

def update_camera_LED():
    logging.debug("Starting 'update_camera_LED()' in thread: " + thread_name())
    global THREAD_STOP
    global CAM_ON
    # Loop while THREAD_STOP flag has not been raised.
    while not THREAD_STOP.state():
        if CAM_ON.state():
            solid(pin = configs.camera_pin, duration = 1.0)
        else:
            continue
    logging.debug("Exiting 'update_camera_LED()' in thread: " + thread_name())

def update_battery_LED():
    logging.debug("Starting 'update_battery_LED()' in thread: " + thread_name())
    global THREAD_STOP
    global BATTERY_GOOD
    global BATTERY_LOW
    global BATTERY_CRITICAL
    # Loop while THREAD_STOP flag has not been raised.
    while not THREAD_STOP.state():
        # BATTERY_CRITICAL takes priority.
        if BATTERY_CRITICAL.state():
            blink(pin = configs.battery_pin, delay = 0.1, duration = 3.0)
        elif BATTERY_LOW.state():
            blink(pin = configs.battery_pin, delay = 0.25, duration = 3.0)
        elif BATTERY_GOOD.state():
            solid(pin = configs.battery_pin, duration = 3.0)
        else:
            continue
    logging.debug("Exiting 'update_battery_LED()' in thread: " + thread_name())

def start_LED_threads():
    logging.debug("Starting 'start_LED_threads()' in thread: " thread_name())
    error_thread = threading.Thread(target = update_error_LED, args = ()).start()
    power_thread = threading.Thread(target = update_power_LED, args = ()).start()
    camera_thread = threading.Thread(target = update_camera_LED, args = ()).start()
    battery_thread = threading.Thread(target = update_battery_LED, args = ()).start()
    logging.debug("LED threads started.")
    global THREADS
    THREADS.append(error_thread)
    THREADS.append(power_thread)
    THREADS.append(camera_thread)
    THREADS.append(battery_thread)
    logging.debug("LED threads appended go global thread list.")
    logging.debug("Exiting 'start_LED_threads()' in thread: " + thread_name())


# EOF
