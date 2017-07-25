import json


def load_configs():
    def byteify(input):
        if isinstance(input, dict):
            return {byteify(key): byteify(value)
                for key, value in input.items()}
        elif isinstance(input, list):
            return [byteify(element) for element in input]
        else:
            return input

    with open("configs.json", "r") as f:
        return byteify(json.load(f))


def main():
    configs = load_configs()

    global class_mode

    nb_classes = configs["global"]["nb_classes"]
    img_width = configs["global"]["img_width"]
    img_height = configs["global"]["img_height"]
    model_save_name = configs["global"]["model_save_name"]

    shear_range = configs["preprocessing"]["shear_range"]
    zoom_range = configs["preprocessing"]["zoom_range"]
    zca_whitening = configs["preprocessing"]["zca_whitening"]
    rotation_range = configs["preprocessing"]["rotation_range"]
    width_shift_range = configs["preprocessing"]["width_shift_range"]
    height_shift_range = configs["preprocessing"]["height_shift_range"]
    vertical_flip = configs["preprocessing"]["vertical_flip"]
    horizontal_flip = configs["preprocessing"]["horizontal_flip"]
    color_mode = configs["preprocessing"]["color_mode"]
    class_mode = configs["preprocessing"]["class_mode"]
    test_fname = configs["preprocessing"]["test_fname"]
    fill_mode = configs["preprocessing"]["fill_mode"]
    nb_test_images = configs["preprocessing"]["nb_test_images"]
    test_dir = configs["preprocessing"]["test_dir"]
    val_fname = configs["preprocessing"]["val_fname"]
    nb_val_images = configs["preprocessing"]["nb_val_images"]
    val_dir = configs["preprocessing"]["val_dir"]


    batch_size = configs["training"]["batch_size"]
    val_batch_size = configs["training"]["val_batch_size"]
    nb_epoch = configs["training"]["nb_epoch"]
    verbose = configs["training"]["verbose"]
    print_summary = configs["training"]["print_summary"]

    vgg16_top_model_weights_path = configs["pretrained"]["vgg16_top_model_weights_path"]
    vgg16_weights_path = configs["pretrained"]["vgg16_weights_path"]

    max_evals = configs["hyperas"]["max_evals"]

main()


if __name__ == '__main__':
    main()
