from lvp import LeanVisionProcessor

process = LeanVisionProcessor()
process.set_test_case()
process.set_alarm_case('dumbbell')
process.set_file_save_case('mask')
process.start_capture()
