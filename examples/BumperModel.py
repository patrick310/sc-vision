from lvp import LeanVisionProcessor
from vdp import VisionDataProcessor


def deploy():
    process = LeanVisionProcessor()
    process.set_model('bumperModel.h5')
    process.set_file_save_case('1bolt_inner')
    process.set_file_save_case('1bolt_outer')
    process.set_file_save_case('2bolt')
    process.start_capture()

def train():
    process = VisionDataProcessor()
    process.create_simple_categorical_model()
    process.fit_model()
    process.save_model_to_file()
    process.plot_model_history()

if __name__ == '__main__':
    train()
