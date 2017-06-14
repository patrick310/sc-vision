import cv2
from keras.models import load_model
import numpy as np
from PIL import Image
import configs
import emails

def set_resolution(cap, x, y):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(x))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(y))
    return str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def format_image_for_network(image):
    image = Image.fromarray(image).resize((configs.img_width,configs.img_height))
    np_frame = np.expand_dims(np.asarray(image), axis=0)
    return np_frame

def send_email_alert(message, frame):
    eml = emails.html(html=message,
                      subject="An error has been detected",
                      mail_from=('Ponder', 'ponder@daimler.com'))
    eml.attach(frame,filename='error.jpg')
    r = eml.send(to='patrick.d.weber@daimler.com', smtp={'host': 'aspmx.l.google.com', 'timeout':5})
    assert r.status_code == 250

model = load_model(configs.model_save_name)

class_labels = ['No Person','People']

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
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

    prediction = model.predict(format_image_for_network(frame))
    print("The class was " + str(prediction[0]))
    cv2.putText(frame, "The image shows {}".format(class_labels[int(round(prediction[0][0]))]),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    cv2.imshow("preview", frame)

    if int(round(prediction[0][0])) is 1:
        message = emails.html(html='<p>The system detected an error',
                              subject='Bumper Bolt Error Detected',
                              mail_from=('Patrick Weber', 'patrick.d.weber@daimler.com'),
                              )
        message.attach(data=oframe, filenam='NIOimage.jpg')

    key = cv2.waitKey(20)
    if key == 97:
        cv2.imwrite("newData/pos/" + file_name , oframe)
        counter = counter + 1
    if key == 108:
        cv2.imwrite("newData/neg/" + file_name , oframe)
        counter = counter + 1
    if key == 27: # exit on ESC
        break
cv2.destroyWindow("preview")



