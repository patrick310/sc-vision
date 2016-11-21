from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import h5py
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
import configs
import numpy as np

def data():
    test_data = h5py.File(configs.test_fname, "r")
    test_image_data = test_data.get("image_data")
    test_class_data = test_data.get("class_data")

    val_data = h5py.File(configs.val_fname, "r")
    val_image_data = val_data.get("image_data")
    val_class_data = val_data.get("class_data")

    # make sure not to modify hdf5 files so copy data.
    X_train = np.copy(test_image_data)
    Y_train = np.copy(test_class_data)
    X_test = np.copy(val_image_data)
    Y_test = np.copy(val_class_data)

    test_data.close()
    val_data.close()

    X_train = X_train.reshape(X_train.shape[0], 1, configs.img_height, configs.img_width)
    X_test = X_test.reshape(X_test.shape[0], 1 , configs.img_height, configs.img_width)

    print X_train.shape, Y_train.shape, X_test.shape, Y_test.shape

    return X_train, Y_train, X_test, Y_test

def model(X_train, Y_train, X_test, Y_test):
    input_shape = (1, configs.img_width, configs.img_height)

    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape = input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    if conditional({{choice(["small", "large"])}}) == "large":
        model.add(Convolution2D(32, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(64, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    if conditional({{choice(['extra', 'no_extra'])}}) == 'extra':
        model.add(Dense({{choice([100, 200])}}))
        model.add(Activation('relu'))

    model.add(Dense(60))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense(4))
    model.add(Activation('sigmoid'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer = {{choice(['rmsprop', 'adam', 'sgd'])}},
        #optimizer = 'rmsprop',
        metrics=['accuracy'])
    model.summary()

    model.fit(X_train, Y_train,
        batch_size = configs.batch_size,
        nb_epoch = configs.nb_epoch,
        verbose = configs.verbose,
        validation_data = (X_test, Y_test)
        )
    score, acc = model.evaluate(X_test, Y_test, verbose = 0)
    print("Test accuracy:", acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = data()
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=configs.max_evals,
                                          trials=Trials())
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    best_model.save(configs.model_save_name)
