from keras.models import Sequential
from keras.models import *
from keras.layers import *
import numpy
from Loader import *
from keras.utils import np_utils

def load_data(img_rows, img_cols):
    asd_l = CV2ImageLoader(path = 'infrared_data/asd', img_size = (img_rows, img_cols), gray = True, count  = 10)
    asd_imgs = asd_l.load_images_from_path()
    print "loaded asd"
    sld_l = CV2ImageLoader(path = 'infrared_data/solid', img_size = (img_rows, img_cols), gray = True, count = 10)
    sld_imgs = sld_l.load_images_from_path()
    print "loaded sld"
    std_l = CV2ImageLoader(path = 'infrared_data/standard', img_size = (img_rows, img_cols), gray = True, count = 10)
    std_imgs = std_l.load_images_from_path()
    print "loaded std"
    bck_l = CV2ImageLoader(path = 'infrared_data/background', img_size = (img_rows, img_cols), gray = True, count = 10)
    bck_imgs = bck_l.load_images_from_path()
    print "loaded bck"
    y = [0 for _ in range(len(asd_imgs))] \
        + [1 for _ in range(len(sld_imgs))] \
        + [2 for _ in range(len(std_imgs))] \
        + [3 for _ in range(len(bck_imgs))]

    mrgr = Merger([asd_l, sld_l, std_l, bck_l])
    imgs_l = mrgr.merge()

    return (imgs_l.get_images(), y)

print "setting variables"
np.random.seed(1337)

batch_size = 15
nb_classes = 4
nb_epoch = 20
img_rows = 150
img_cols = 150

nb_filters = 30
pool_size = (5, 5)
kernel_size = (8, 8)

print "loading images"

(x, y) = load_data(img_rows, img_cols)

print "formatting loaded data"

if K.image_dim_ordering() == 'th':
    x = x.reshape(x.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x = x.reshape(x.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x = x.astype('float32')
x /= 255
print 'x shape:', x.shape
print x.shape[0], 'training samples'

y = np_utils.to_categorical(y, nb_classes)

print "building network"

model = Sequential()
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(400))
model.add(Activation('relu'))
model.add(Dense(200))
model.add(Dropout(0.25))
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

print "compiling"

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

print "training"

model.fit(x, y, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1)

#model.save('mymodel.h5')
