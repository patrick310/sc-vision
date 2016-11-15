import time
import datetime

def now():
    return datetime.datetime.fromtimestamp(time.time())

class Timer:
    def __init__(self):
        self.time = None

    def start(self):
        self.time = time.time()
