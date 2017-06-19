# -*- coding: utf-8 -*-
from . import helpers
from . import vision_data_processor


def train():
    """Contemplation..."""
    dataprocessor = vision_data_processor.VisionDataProcessor()
    dataprocessor.create_simple_categorical_model()
    #dataprocessor.create_binary_vgg16_model()
    #dataprocessor.create_simple_binary_model()
    #dataprocessor.create_flat_binary_fc_model()
    #dataprocessor.create_doe_model()
    #dataprocessor.create_flat_keras_model()
    #dataprocessor.inception_cross_train()
    dataprocessor.fit_model()
    dataprocessor.plot_model_history()