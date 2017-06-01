from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.applications.inception_v3 import InceptionV3
from keras import optimizers
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional

from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
import h5py
import configs
import numpy as np
import os

class VisionDataProcessor():

    def __init__(self):
        self.training_data_generator = ImageDataGenerator(
        shear_range = configs.shear_range,
        zoom_range = configs.zoom_range,
        zca_whitening = configs.zca_whitening,
        rotation_range = configs.rotation_range,
        width_shift_range = configs.width_shift_range,
        height_shift_range = configs.height_shift_range,
        vertical_flip = configs.vertical_flip,
        horizontal_flip = configs.horizontal_flip,
        rescale=1./255,
        fill_mode = 'nearest',
        )

        self.input_shape = (configs.img_width, configs.img_height, 3)

        self.validation_data_generator = ImageDataGenerator()
        
        self.train_generator = self.create_data_generator_from_directory(configs.test_dir, 'train')
        self.validation_generator = self.create_data_generator_from_directory(configs.val_dir, 'validate')
    
    def create_data_generator_from_directory(self, directory, generator):
        if generator == "train":
            generated_generator = self.training_data_generator.flow_from_directory(
            directory = directory,
            target_size = (configs.img_width, configs.img_height),
            batch_size = configs.batch_size,
            color_mode = configs.color_mode,
            class_mode = configs.class_mode,
            )
        elif generator == "validate":
            generated_generator = self.validation_data_generator.flow_from_directory(
                directory = directory,
                target_size= (configs.img_width, configs.img_height),
                batch_size=configs.batch_size,
                color_mode=configs.color_mode,
                class_mode=configs.class_mode,
            )
        else:
            generated_generator = None
        
        return generated_generator
            
    def generate_h5py_files(self):
        test_file = h5py.File(configs.test_fname, "w")
        self.test_image_data = test_file.create_dataset(
            "image_data",
            (configs.nb_test_images, configs.img_width, configs.img_height),
            dtype = "float32"
        )
        self.test_class_data = test_file.create_dataset(
            "class_data",
            (configs.nb_test_images, configs.nb_classes),
            dtype = "float32"
        )

        val_file = h5py.File(configs.val_fname, "w")
        self.val_image_data = val_file.create_dataset(
            "image_data",
            (configs.nb_val_images, configs.img_width, configs.img_height),
            dtype = 'float32'
        )
        self.val_class_data = val_file.create_dataset(
            "class_data",
            (configs.nb_val_images, configs.nb_classes),
            dtype = "float32"
        )
       
    def load_h5py_with_generator(self,counter_limit,data_generator):
        #Can't figure out how to remove the extra dimension
        print(type(self.validation_generator))
        counter = 0
        while counter != counter_limit:
            data_pack = data_generator.next() # tuple
            image_data = data_pack[0]
            class_data = data_pack[1]
            

            for index in range(len(image_data)):
                print(image_data[index][0].shape)
                print(self.test_image_data[counter][0].shape)
                self.test_image_data[counter] = image_data[index]
                self.test_class_data[counter] = class_data[index]
                counter += 1
                if counter == configs.nb_test_images:
                    break
            import sys
            sys.stdout.write("  " + str(counter) + "/" + str(configs.nb_test_images) + "\r")
            sys.stdout.flush()
        
    def create_simple_shallow_binary_model(self):
        
        model = Sequential()
         
        model.add(Conv2D(24, (3, 3),
            padding='same',
            data_format='channels_last',
            strides=2,
            input_shape=self.input_shape))

        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(4, 4)))
        
        model.add(Flatten())
        model.add(Dense(75,activation='relu'))
        model.add(Dense(40,activation='relu'))
        model.add(Dense(10,activation='relu'))

        model.add(Dropout(0.5))
        
        model.add(Dense(configs.nb_classes, activation='sigmoid'))
        #model.add(Activation('sigmoid'))
            
        if configs.print_summary:
            model.summary()

        sgd = optimizers.SGD(lr=0.05, decay=1e-6, momentum=1.1, nesterov=True)

        model.compile(optimizer='Nadam',
              loss='binary_crossentropy',
              metrics=['binary_accuracy'])
              
        self.model = model

    def create_flat_keras_model(self):
        #I want this to essentially be a linear/logistic regression

        model = Sequential()

        model.add(Conv2D(12, (2, 2),
                         padding='same',
                         data_format='channels_last',
                         input_shape=self.input_shape))

        model.add(Activation('relu'))

        model.add(Flatten())
        # now: model.output_shape == (None, 64, 32, 32)

        model.add(Dense(configs.nb_classes))

        if configs.print_summary:
            model.summary()
        model.compile(optimizer='sgd',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        self.model = model

    def create_other_doe_model(self):
        model = Sequential()
        model.add(Dense(512, input_shape=self.input_shape))
        model.add(Activation('relu'))
        model.add(Dropout({{uniform(0, 1)}}))
        model.add(Dense({{choice([256, 512, 1024])}}))
        model.add(Activation({{choice(['relu', 'sigmoid'])}}))
        model.add(Dropout({{uniform(0, 1)}}))

        if conditional({{choice(['three', 'four'])}}) == 'four':
            model.add(Dense(100))

            # We can also choose between complete sets of layers

            model.add({{choice([Dropout(0.5), Activation('linear')])}})
            model.add(Activation('relu'))

        model.add(Dense(10))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                      optimizer={{choice(['rmsprop', 'adam', 'sgd'])}})

    def create_doe_model(self):
        model = Sequential()
        model.add(Convolution2D(32, 3, 3, input_shape=self.input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        if conditional({{choice(['three', 'four'])}}) == 'four':
            model.add(Convolution2D(32, 3, 3))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Convolution2D(64, 3, 3, input_shape=self.input_shape))
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

        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

        model.compile(
            loss='categorical_crossentropy',
            optimizer={{choice(['rmsprop', 'adam', sgd])}},
            metrics=['accuracy'])

        if configs.print_summary:
            model.summary()

        self.model = model

        self.model.fit_generator(
            self.train_generator,
            steps_per_epoch=int(configs.nb_test_images/configs.batch_size),
            epochs=configs.nb_epoch,
            validation_data=self.validation_generator,
            validation_steps=int(configs.nb_val_images/configs.batch_size)
            )

        return {'loss': -acc, 'status': STATUS_OK, 'model': self.create_doe_model}

    def fit_simple_keras_model(self):
        self.model.fit_generator(
            self.train_generator,
            steps_per_epoch=int(configs.nb_test_images/configs.batch_size),
            epochs=configs.nb_epoch,
            validation_data=self.validation_generator, 
            validation_steps=int(configs.nb_val_images/configs.batch_size)
            )

    def fit_inception_model(self):
        self.inception_model.fit_generator(
            self.train_generator,
            steps_per_epoch=int(configs.nb_test_images/configs.batch_size),
            epochs=configs.nb_epoch,
            validation_data=self.validation_generator,
            validation_steps=int(configs.nb_val_images/configs.batch_size)
            )

    def fit_inception_model_short(self):
        self.inception_model.fit_generator(
            self.train_generator,
            steps_per_epoch=int(configs.nb_test_images/configs.batch_size),
            epochs=5,
            validation_data=self.validation_generator,
            validation_steps=int(configs.nb_val_images/configs.batch_size)
            )

    def save_trained_keras_model_to_file(self):
        self.model.save(configs.model_save_name)
        return None

    def inception_cross_train(self):
        # create the base pre-trained model
        input_tensor = Input(shape=(configs.img_width, configs.img_height, 3))
        #base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=True)
        base_model = InceptionV3(weights='imagenet', include_top=False)

        # add a global spatial average pooling layer
        base_model.summary()
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        # and a logistic layer -- let's say we have 200 classes
        predictions = Dense(2, activation='sigmoid')(x)

        # this is the model we will train
        self.inception_model = Model(inputs=base_model.input, outputs=predictions)

        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in base_model.layers:
            layer.trainable = False

        # compile the model (should be done *after* setting layers to non-trainable)
        self.inception_model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

        # train the model on the new data for a few epochs
        self.fit_inception_model_short()

        # at this point, the top layers are well trained and we can start fine-tuning
        # convolutional layers from inception V3. We will freeze the bottom N layers
        # and train the remaining top layers.

        # let's visualize layer names and layer indices to see how many layers
        # we should freeze:
        for i, layer in enumerate(base_model.layers):
            print(i, layer.name)

        # we chose to train the top 2 inception blocks, i.e. we will freeze
        # the first 172 layers and unfreeze the rest:
        for layer in model.layers[:172]:
            layer.trainable = False
        for layer in model.layers[172:]:
            layer.trainable = True

        # we need to recompile the model for these modifications to take effect
        # we use SGD with a low learning rate
        from keras.optimizers import SGD
        model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

        # we train our model again (this time fine-tuning the top 2 inception blocks
        # alongside the top Dense layers
        self.fit_inception_model()
        
if __name__ == '__main__':
    dataprocessor = VisionDataProcessor()
    dataprocessor.create_simple_shallow_binary_model()
    #dataprocessor.create_doe_model()
    #dataprocessor.create_flat_keras_model()
    #dataprocessor.inception_cross_train()
    dataprocessor.fit_simple_keras_model()
    dataprocessor.save_trained_keras_model_to_file()
