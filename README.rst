SC-Vision
========================

This utility will assist in preparing, testing, and deploying vision system models.

Basic Usage:

To train a model, create a folder holding your images (separate into folders by class). Paste the configs template [from the github] into the folder. In the interpreter, type:

vdp.static()



Configs template:

{
  "global" : {
    "nb_classes" : 4,
    "img_width" : 600,
    "img_height" : 600,
    "model_save_name" : "model_output.h5"
  },

  "preprocessing" : {
    "shear_range" : 0,
    "zoom_range" : 0.2,
    "zca_whitening" : false,
    "rotation_range" : 5,
    "width_shift_range" : 0.05,
    "height_shift_range" : 0.0,
    "vertical_flip" : false,
    "horizontal_flip" : false,
    "color_mode" : "rgb",
    "class_mode" : "categorical",
    "fill_mode" : "nearest",
    "test_fname" : "test_data.hdf5",
    "nb_test_images" : 1000,
    "test_dir" : "C://Users//patri//PycharmProjects//sc-vision//vision//train//INRIAPerson//Train",
    "val_fname"  : "val_data.hdf5",
    "nb_val_images" : 200,
    "val_dir" : "C://Users//patri//PycharmProjects//sc-vision//vision//train//INRIAPerson//Test"
  },

  "training" : {
    "batch_size" : 50,
    "val_batch_size" : 1,
    "nb_epoch"   : 50,
    "verbose" : true,
    "print_summary" : true
  },

  "pretrained" : {
    "vgg16_weights_path" : "vgg16_weights.h5",
    "vgg16_top_model_weights_path" : "bottleneck_fc_model.h5"
  },


  "hyperas" : {
    "max_evals" : 25
  }
}






`Learn more <http://www.kennethreitz.org/essays/repository-structure-and-python>`_.
