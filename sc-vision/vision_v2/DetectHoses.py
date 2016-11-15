from VisionResources.imageprocessing import *

im = open_image_from_file("/home/mbusi/program_master_dir/sc-vision/sc-vision/vision_v2/demos/ImageProcessorDemo/original.jpg")
arr = PIL_to_numpy(im)
