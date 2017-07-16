# -*- coding: utf-8 -*-
from . import helpers
from . import vision_data_processor


def train():
    """Contemplation..."""
    dataprocessor = vision_data_processor.VisionDataProcessor()
    dataprocessor.create_simple_categorical_model()
    dataprocessor.plot_model_history()


#This file needs to hold the arg parser where we will specify which model to use