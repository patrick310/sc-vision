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


def log_new_car(log_file):
    log_file.write("[Info]  " + datetime.now().strftime(configs.time_format) + " New vehicle detected\n")


def log_car_configuration(log_file, bolt_configuration):
    log_file.write("[Info]  " + datetime.now().strftime(configs.time_format) + " Vehicle identified with ")
    log_file.write(bolt_configuration)
    log_file.write(" pattern\n")


def log_car_leaving(log_file):
    log_file.write("[Info]  " + datetime.now().strftime(configs.time_format) + " Vehicle detected leaving station\n")
	

def log_message(log_file, message):
	log_file.write(message)

