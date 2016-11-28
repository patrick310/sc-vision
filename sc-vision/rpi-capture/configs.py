import json

def load_configs():
    def byteify(input):
        if isinstance(input, dict):
            return {byteify(key): byteify(value)
                for key, value in input.iteritems()}
        elif isinstance(input, list):
            return [byteify(element) for element in input]
        elif isinstance(input, unicode):
            return input.encode('utf-8')
        else:
            return input

    with open("configs.json", "r") as f:
        return byteify(json.load(f))

configs = load_configs()

img_width = configs["img_size"]["width"]
img_height = configs["img_size"]["height"]

time_limit_enabled = configs["time_limit"]["enabled"]
time_limit_hours = configs["time_limit"]["hours"]
time_limit_minutes = configs["time_limit"]["minutes"]
time_limit_seoncds = configs["time_limit"]["seconds"]

image_limit_enabled = configs["image_limit"]["enabled"]
image_limit_count = configs["image_limit"]["count"]

GPIO_enabled = configs["GPIO"]["enabled"]
power_pin = configs["GPIO"]["power"]
camera_pin = configs["GPIO"]["camera"]
error_pin = configs["GPIO"]["error"]
battery_pin = configs["GPIO"]["battery"]
buzzer_pin = configs["GPIO"]["buzzer"]
light_ring_pin = configs["GPIO"]["light_ring"]

model_filepath = configs["model_filepath"]
image_descriptor = configs["image_descriptor"]

save_folder = configs["save_folder"]

capture_delay = configs["capture_delay"]

nb_classes = configs["nb_classes"]
