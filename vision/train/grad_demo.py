import cv2
from keras.models import load_model
from vis.visualization import visualize_saliency
import numpy as np
from PIL import Image

from keras.preprocessing.image import img_to_array
import configs

from vis.utils import utils
from vis.utils.vggnet import VGG16
from vis.visualization import visualize_saliency
from vis.visualization import visualize_cam

def set_resolution(cap, x, y):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(x))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(y))
    return str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


def format_image_for_network(image):
    image = Image.fromarray(image).resize((configs.img_width, configs.img_height))
    np_frame = np.expand_dims(np.asarray(image), axis=0)
    return np_frame


model = load_model(configs.model_save_name)
model = VGG16(weights='imagenet', include_top=True)
print('Model loaded.')


layer_name = 'predictions'
layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]
cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False


counter = 0

set_resolution(vc, 224, 224)

while rval:
    rval, frame = vc.read()

    #seed_img = utils.load_img(frame, target_size=(224, 224))
    seed_img = frame
    # Convert to BGR, create input with batch_size: 1, and predict.

    bgr_img = utils.bgr2rgb(seed_img)

    img_input = format_image_for_network(frame)

    pred_class = np.argmax(model.predict(img_input))

    heatmap = visualize_cam(model, layer_idx, [pred_class], seed_img)

    if show:
        plt.axis('off')
        plt.imshow(heatmap)
        plt.title('Attention - {Person}')
        plt.show()
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break

cv2.destroyWindow("preview")



