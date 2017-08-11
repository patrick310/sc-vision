import json


class ConfigManager:

    def __init__(self, source=None):

        if source is 'file':
            self.load_parameters_from_file()

        elif source is 'dictionary':
            # To-do but would make the constructor super long
            self.load_parameters_from_dictionary()

        else:
            self.load_default_parameters()

    def load_parameters_from_file(self):

        def load_configs_file(self, file_name=None):
            if file_name is None:
                file_name = "configs.json"

            def byteify(input):
                if isinstance(input, dict):
                    return {byteify(key): byteify(value)
                            for key, value in input.items()}
                elif isinstance(input, list):
                    return [byteify(element) for element in input]
                else:
                    return input

            with open(file_name, "r") as f:
                return byteify(json.load(f))

        configs = load_configs_file()

        self.nb_classes = configs["global"]["nb_classes"]
        self.img_width = configs["global"]["img_width"]
        self.img_height = configs["global"]["img_height"]
        self.model_save_name = configs["global"]["model_save_name"]

        self.shear_range = configs["preprocessing"]["shear_range"]
        self.zoom_range = configs["preprocessing"]["zoom_range"]
        self.zca_whitening = configs["preprocessing"]["zca_whitening"]
        self.rotation_range = configs["preprocessing"]["rotation_range"]
        self.width_shift_range = configs["preprocessing"]["width_shift_range"]
        self.height_shift_range = configs["preprocessing"]["height_shift_range"]
        self.vertical_flip = configs["preprocessing"]["vertical_flip"]
        self.horizontal_flip = configs["preprocessing"]["horizontal_flip"]
        self.color_mode = configs["preprocessing"]["color_mode"]
        self.class_mode = configs["preprocessing"]["class_mode"]
        self.fill_mode = configs["preprocessing"]["fill_mode"]
        self.nb_test_images = configs["preprocessing"]["nb_test_images"]
        self.test_dir = configs["preprocessing"]["test_dir"]
        self.nb_val_images = configs["preprocessing"]["nb_val_images"]
        self.val_dir = configs["preprocessing"]["val_dir"]

        self.batch_size = configs["training"]["batch_size"]
        self.val_batch_size = configs["training"]["val_batch_size"]
        self.nb_epoch = configs["training"]["nb_epoch"]
        self.verbose = configs["training"]["verbose"]
        self.print_summary = configs["training"]["print_summary"]

        self.vgg16_top_model_weights_path = configs["pretrained"]["vgg16_top_model_weights_path"]
        self.vgg16_weights_path = configs["pretrained"]["vgg16_weights_path"]

    def load_default_parameters(self):
        self.nb_classes = 2
        self.img_width = 200
        self.img_height = 200
        self.model_save_name = 'myModel.h5'

        self.shear_range = 0
        self.zoom_range = 0
        self.zca_whitening = False
        self.rotation_range = 0
        self.width_shift_range = 0
        self.height_shift_range = 0
        self.vertical_flip = False
        self.horizontal_flip = False
        self.color_mode = 'rgb'
        self.class_mode = 'categorical'
        self.fill_mode = 'nearest'
        self.nb_test_images = 7000
        self.train_dir = 'dc_train_data/train'
        self.nb_val_images = 2000
        self.val_dir = 'dc_train_data/validate'

        self.batch_size = 15
        self.val_batch_size = 5
        self.nb_epoch = 12
        self.verbose = 1
        self.print_summary = True

        self.vgg16_top_model_weights_path = "reference/bottleneck_fc_model.h5"
        self.vgg16_weights_path = "reference/vgg16_weights.h5"

    def load_parameters_from_dictionary(self):
        None
