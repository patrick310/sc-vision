import cv2
from keras.models import load_model
import numpy as np
from PIL import Image
import configs
from attention_maps import generate_cam_from_image


def set_resolution(cap, x, y):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(x))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(y))
    return str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


def format_image_for_network(image):
    image = Image.fromarray(image).resize((configs.img_width, configs.img_height))
    np_frame = np.expand_dims(np.asarray(image), axis=0)
    return np_frame


model = load_model(configs.model_save_name)

class_labels = ['No Person', 'People']

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

counter = 0
prediction = "debug"

set_resolution(vc, 1920, 1080)

while rval:
    file_name = str(counter) + ".jpg"
    rval, frame = vc.read()
    oframe = frame.copy()
    frame = generate_cam_from_image(frame,model,layer=15)

    prediction = model.predict(format_image_for_network(oframe))
    print("The class was " + str(prediction[0]))
    cv2.putText(oframe, "The image shows {}".format(class_labels[int(round(prediction[0][0]))]),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    frame = np.asarray(frame)
    print(type(frame))
    cv2.imshow("preview", frame)
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break
cv2.destroyWindow("preview")



