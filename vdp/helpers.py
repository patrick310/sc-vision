def get_answer():
    """Get an answer."""
    return True

def create_model_config():
    """Creates a functioning config file for testing purposes"""
    return True


def set_resolution(cap, x, y):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(x))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(y))
    return str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


def format_image_for_network(image):
    image = Image.fromarray(image).resize((configs.img_width, configs.img_height))
    np_frame = np.expand_dims(np.asarray(image), axis=0)
    return np_frame


def log_new_car():
    F.write("[Info]  " + datetime.now().strftime(configs.time_format) + " New vehicle detected\n")


def log_car_configuration(bolt_configuration):
    F.write("[Info]  " + datetime.now().strftime(configs.time_format) + " Vehicle identified with ")
    F.write(bolt_configuration)
    F.write(" pattern\n")


def log_car_leaving():
    F.write("[Info]  " + datetime.now().strftime(configs.time_format) + " Vehicle detected leaving station\n")

