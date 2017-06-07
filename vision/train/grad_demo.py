import cv2
from keras.models import load_model
import numpy as np
from PIL import Image
import configs
from gradcam import grad_activation_map

def set_resolution(cap, x, y):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(x))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(y))
    return str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


def format_image_for_network(image):
    image = Image.fromarray(image).resize((configs.img_width, configs.img_height))
    np_frame = np.expand_dims(np.asarray(image), axis=0)
    return np_frame


model = load_model(configs.model_save_name)

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False
    
print("frame type", type(frame))

counter = 0
prediction = "debug"

set_resolution(vc, 224, 224)

while rval:
    rval, frame = vc.read()
    oframe = frame.copy()
    try:
        grad_image = Image.fromarray(grad_activation_map(format_image_for_network(frame)))
        cv2.imshow("preview", grad_image)
        print('grad image ', type(grad_image))
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break
    except ValueError:
        print("Value Error",)
cv2.destroyWindow("preview")



