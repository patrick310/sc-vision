#This script takes an input of vectorized images and returns a trained model + model performance metrics

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

#[TODO] Ensure the create_data module outputs a single hypy set (image and class data)
#[TODO] Use scikitlearn test_train split to split the data in train_network module
#[TODO] Remove all requirements to hard-code image counts
#[TODO] Add test_train split ratio to config file
#[NOTE] C and E can delete this to demonstrate GitHub knowledge!!

def data():
	#Takes h5py compacted data and returns x and y test and train numpy arrays
    test_data = h5py.File(configs.test_fname, "r")
    test_image_data = test_data.get("image_data")
    test_class_data = test_data.get("class_data")

    val_data = h5py.File(configs.val_fname, "r")
    val_image_data = val_data.get("image_data")
    val_class_data = val_data.get("class_data")

    # make sure not to modify hdf5 files so copy data.
	#[NOTE] Right now we are loading the entire dataset to memory
	#[TODO] Look into improving and "streaming" from file 
    X_train = np.copy(test_image_data)
    Y_train = np.copy(test_class_data)
    X_test = np.copy(val_image_data)
    Y_test = np.copy(val_class_data)

    test_data.close()
    val_data.close()
	
	#[NOTE] This may be a redundant image re-size
    X_train = X_train.reshape(X_train.shape[0], 1, configs.img_height, configs.img_width)
    X_test = X_test.reshape(X_test.shape[0], 1 , configs.img_height, configs.img_width)

    return X_train, Y_train, X_test, Y_test

def model(X_train, Y_train, X_test, Y_test):
	#First, defines Keras model structure. The script then fits the model to the data in the parameters and returns metrics and the model
	#[TODO] Break the two distinct steps into two functions... model intialization and model fit
	#[TODO] Come up with better names for the functions
	#[TODO] Visualize model structure decision tree - ensure conditions are useful. Choose the splits with the most impact.
	
    input_shape = (1, configs.img_width, configs.img_height)

    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape = input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    if conditional({{choice(["small", "large"])}}) == "large":
        model.add(Convolution2D(32, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(64, 3, 3, input_shape = input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense({{choice([75, 100])}}))
    model.add(Activation('relu'))

    if conditional({{choice(['extra', 'no_extra'])}}) == 'extra':
        model.add(Dense(50))
        model.add(Activation('relu'))

    model.add(Dense({{choice([10, 20, 30])}}))
    model.add(Activation('relu'))

    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense(configs.nb_classes))
    model.add(Activation('sigmoid'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer = {{choice(['rmsprop', 'adam', 'sgd'])}},
        metrics=['accuracy'])

    if configs.print_summary:
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
	#[TODO] Follow this flow through
    X_train, Y_train, X_test, Y_test = data()
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=configs.max_evals,
                                          trials=Trials())
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    best_model.save(configs.model_save_name)
