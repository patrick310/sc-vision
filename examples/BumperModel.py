from lvp import LeanVisionProcessor

process = LeanVisionProcessor()
process.set_model('bumperModel.h5')
process.set_file_save_case('1bolt_inner')
process.set_file_save_case('1bolt_outer')
process.set_file_save_case('2bolt')
process.start_capture()
