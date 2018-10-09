from vdp.vision_data_processor import VisionDataProcessor


def train():
    """Contemplation..."""
    dataprocessor = VisionDataProcessor()
    dataprocessor.create_simple_categorical_model()
    dataprocessor.plot_model_history()

class VisionDataProcessor(VisionDataProcessor):
    '''
    A class representing a training process
    '''
    None

#This file needs to hold the arg parser where we will specify which model to use