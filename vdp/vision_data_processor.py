from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers import Activation, Dropout, Flatten, Dense, ZeroPadding2D, MaxPooling2D, Input
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.applications.vgg16 import VGG16
from keras.models import Model, load_model
from keras.utils import to_categorical

import matplotlib.pyplot as plt
import numpy as np

from vdp.configs import ConfigManager
import vdp.helpers


class VisionDataProcessor:

    def __init__(self):
        self.configs = ConfigManager()

        self.training_data_generator = ImageDataGenerator(
            shear_range = self.configs.shear_range,
            zoom_range = self.configs.zoom_range,
            zca_whitening= self.configs.zca_whitening,
            rotation_range= self.configs.rotation_range,
            width_shift_range = self.configs.height_shift_range,
            vertical_flip = self.configs.vertical_flip,
            horizontal_flip = self.configs.horizontal_flip,
            rescale=1./255,
            fill_mode = self.configs.fill_mode,
        )

        self.input_shape = (self.configs.img_width, self.configs.img_height, 3)

        self.validation_data_generator = ImageDataGenerator()
        
        self.train_generator = self.create_data_generator_from_directory(directory=self.configs.train_dir,
                                                                         generator='train')
        self.validation_generator = self.create_data_generator_from_directory(directory=self.configs.val_dir,
                                                                              generator='validate')

    def create_data_generator_from_directory(self, directory, generator, shuffle=True, class_mode=None):

        if class_mode is None:
            class_mode = self.configs.class_mode

        if generator == "train":
            generated_generator =\
                self.training_data_generator.flow_from_directory(
                    directory = directory,
                    target_size = (self.configs.img_width, self.configs.img_height),
                    batch_size = self.configs.batch_size,
                    color_mode = self.configs.color_mode,
                    class_mode= class_mode,
                    shuffle=shuffle,
                )
        elif generator == "validate":
            generated_generator = self.validation_data_generator.flow_from_directory(
                directory=directory,
                target_size=(self.configs.img_width, self.configs.img_height),
                batch_size= self.configs.batch_size,
                color_mode= self.configs.color_mode,
                class_mode=class_mode,
                shuffle=shuffle,
            )
        else:
            generated_generator = None
        
        return generated_generator

    def create_binary_vgg16_model(self):

        # https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
        def save_bottleneck_features():

            # build the VGG16 network
            self.model = applications.VGG16(include_top=False, weights='imagenet')

            self.ordered_train_generator = \
                self.create_data_generator_from_directory(
                    self.configs.test_dir,
                    'train',
                    False,
                    class_mode=None
                )

            bottleneck_features_train = self.model.predict_generator(
                self.ordered_train_generator, self.configs.nb_test_images // self.configs.batch_size)
            np.save(open('bottleneck_features_train.npy', 'wb'),
                    bottleneck_features_train)

            self.ordered_validation_generator =\
                self.create_data_generator_from_directory(
                    self.configs.val_dir,
                    'validate',
                    False,
                    class_mode=None
                )

            bottleneck_features_validation = self.model.predict_generator(
                self.ordered_validation_generator, self.configs.nb_val_images // self.configs.batch_size)
            np.save(open('bottleneck_features_validation.npy', 'wb'),
                    bottleneck_features_validation)

        def train_top_model():
            train_data = np.load(open('bottleneck_features_train.npy', 'rb'))
            train_labels = np.array(
                [0] * (self.configs.nb_test_images // 2) + [1] * (self.configs.nb_test_images // 2))

            validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))
            validation_labels = np.array(
                [0] * (self.configs.nb_val_images // 2) + [1] * (self.configs.nb_val_images // 2))

            def create_flat_binary_fc_model(input_shape=None):
                if input_shape is None:
                    input_shape = self.input_shape
                model = Sequential()
                model.add(Flatten(input_shape=input_shape))
                model.add(Dense(256, activation='relu'))
                model.add(Dropout(0.5))
                model.add(Dense(1, activation='sigmoid'))

                model.compile(optimizer='rmsprop',
                              loss='binary_crossentropy',
                              metrics=['accuracy'])

                self.model = model

            create_flat_binary_fc_model(train_data.shape[1:])

            self.model.fit(train_data, train_labels,
                           epochs= self.configs.nb_epoch,
                           batch_size= self.configs.batch_size,
                           validation_data=(validation_data, validation_labels)
                           )

            self.model.save_weights(self.configs.vgg16_top_model_weights_path)

        def fine_tune_top_model():

            # build the VGG16 network
            base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape)
            print('Model loaded.')

            # build a classifier model to put on top of the convolutional model
            top_model = Sequential()
            top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
            top_model.add(Dense(256, activation='relu'))
            top_model.add(Dropout(0.5))
            top_model.add(Dense(1, activation='sigmoid',name='predictions'))

            # note that it is necessary to start with a fully-trained
            # classifier, including the top classifier,
            # in order to successfully do fine-tuning
            top_model.load_weights(self.configs.vgg16_top_model_weights_path)

            # add the model on top of the convolutional base
            model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

            # set the first 155 layers (up to the last conv block)
            # to non-trainable (weights will not be updated)
            for layer in model.layers[:15]:
                layer.trainable = False

            # compile the model with a SGD/momentum optimizer
            # and a very slow learning rate.
            model.compile(loss='binary_crossentropy',
                          optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                          metrics=['accuracy'])

            self.model = model

        save_bottleneck_features()
        train_top_model()
        fine_tune_top_model()
        self.fit_model()

    def conv2DReluBatchNorm(self, n_filter, w_filter, h_filter, inputs):
        return BatchNormalization()(
            Activation(activation='relu')(Conv2D(n_filter, w_filter, h_filter, border_mode='same')(inputs)))

    def create_categorical_vgg16_model(self):

        # https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
        def save_bottleneck_features():

            # build the VGG16 network
            base_model = applications.VGG16(include_top=False, weights='imagenet', pooling='avg')

            top_model = Sequential()
            top_model.add(Dense(self.configs.nb_classes, activation='softmax', input_shape=base_model.output_shape[1:]))
            self.model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

            self.ordered_train_generator = \
                self.create_data_generator_from_directory(
                    self.configs.test_dir,
                    'train',
                    False,
                    class_mode=None
                )

            self.bottleneck_features_train = self.model.predict_generator(
                self.ordered_train_generator, self.configs.nb_test_images // self.configs.batch_size)

            self.train_labels = to_categorical(self.ordered_train_generator.classes.tolist(), num_classes= self.configs.nb_classes)


            self.ordered_validation_generator =\
                self.create_data_generator_from_directory(
                    self.configs.val_dir,
                    'validate',
                    False,
                    class_mode=None
                )

            self.bottleneck_features_validation = self.model.predict_generator(
                self.ordered_validation_generator, self.configs.nb_val_images // self.configs.batch_size)
            self.validation_labels = to_categorical(self.ordered_validation_generator.classes.tolist(), num_classes= self.configs.nb_classes)

        def train_top_model():
            sgd = optimizers.SGD(lr=0.05, decay=1e-6, momentum=1.1, nesterov=True)

            self.model.compile(optimizer=sgd,  # 'Nadam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

            self.model.fit(self.bottleneck_features_train, self.train_labels,
                           epochs= self.configs.nb_epoch,
                           batch_size= self.configs.batch_size,
                           validation_data=(self.bottleneck_features_validation, self.validation_labels)
                           )

            self.model.save_weights(self.configs.vgg16_top_model_weights_path)

        def fine_tune_top_model():

            # build the VGG16 network
            base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape)
            print('Model loaded.')

            # build a classifier model to put on top of the convolutional model
            top_model = Sequential()
            top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
            top_model.add(Dense(256, activation='relu'))
            top_model.add(Dropout(0.5))
            top_model.add(Dense(1, activation='sigmoid',name='predictions'))

            # note that it is necessary to start with a fully-trained
            # classifier, including the top classifier,
            # in order to successfully do fine-tuning
            top_model.load_weights(self.configs.vgg16_top_model_weights_path)

            # add the model on top of the convolutional base
            model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

            # set the first 155 layers (up to the last conv block)
            # to non-trainable (weights will not be updated)
            for layer in model.layers[:15]:
                layer.trainable = False

            # compile the model with a SGD/momentum optimizer
            # and a very slow learning rate.
            model.compile(loss='binary_crossentropy',
                          optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                          metrics=['accuracy'])

            self.model = model

        save_bottleneck_features()
        train_top_model()
        #fine_tune_top_model()
        #self.fit_model()

    def create_vgg16_model(self):

        # Get back the convolutional part of a VGG network trained on ImageNet
        model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
        model_vgg16_conv.summary()

        # Create your own input format (here 3x200x200)
        input = Input(shape=self.input_shape, name='image_input')

        # Use the generated model
        output_vgg16_conv = model_vgg16_conv(input)

        # Add the fully-connected layers
        x = Flatten(name='flatten')(output_vgg16_conv)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(self.configs.nb_classes, activation='softmax', name='predictions')(x)

        # Create your own model
        my_model = Model(input=input, output=x)

        self.model = my_model

        # In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
        my_model.summary()

        sgd = optimizers.SGD(lr=0.05, decay=1e-6, momentum=1.1, nesterov=True)

        self.model.compile(optimizer=sgd,#'Nadam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])


        # Then training with your data !

    def create_simple_binary_model(self):
        
        model = Sequential()
         
        model.add(Conv2D(32, (3, 3),
            padding='same',
            data_format='channels_last',
            strides=1,
            input_shape=self.input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.configs.nb_classes))
        model.add(Activation('sigmoid'))
            
        if self.configs.print_summary:
            model.summary()

        sgd = optimizers.SGD(lr=0.05, decay=1e-6, momentum=1.1, nesterov=True)

        model.compile(optimizer='Nadam',
              loss='binary_crossentropy',
              metrics=['binary_accuracy'])
              
        self.model = model

    def create_simple_categorical_model(self):

        model = Sequential()
        model.add(self.conv2DReluBatchNorm(64,4,4,self.input_shape))
        ''' model.add(Conv2D(64, (4, 4),
                         padding='same',
                         data_format='channels_last',
                         strides=1,
                         input_shape=self.input_shape))'''
        model.add(Activation('relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dense(2028))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2028))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.configs.nb_classes))
        model.add(Activation('softmax'))

        if self.configs.print_summary:
            model.summary()

        sgd = optimizers.SGD(lr=0.05, decay=1e-6, momentum=1.1, nesterov=True)

        model.compile(optimizer=sgd,#'Nadam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        self.model = model

    def create_simple_normalized_model(self):
        # import BatchNormalization


        # instantiate model
        model = Sequential()
        model.add(Conv2D(64, (4, 4),
                         padding='same',
                         data_format='channels_last',
                         strides=1,
                         input_shape=self.input_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        # we can think of this chunk as the hidden layer
        model.add(Dense(64, init='uniform'))
        model.add(Activation('tanh'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Flatten())

        # we can think of this chunk as the output layer
        model.add(Dense(self.configs.nb_classes, init='uniform', activation='softmax'))

        if self.configs.print_summary:
            model.summary()

        sgd = optimizers.SGD(lr=0.05, decay=1e-6, momentum=1.1, nesterov=True)

        model.compile(optimizer='Nadam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        self.model = model

    def fit_model(self):
        history = self.model.fit_generator(
            self.train_generator,
            steps_per_epoch=int(self.configs.nb_test_images/self.configs.batch_size),
            epochs= self.configs.nb_epoch,
            validation_data=self.validation_generator,
            validation_steps=int(self.configs.nb_val_images/self.configs.val_batch_size),
            verbose=1
            )
			

        self.history = history

    def save_model_to_file(self):
        self.model.save(self.configs.model_save_name)
        return None

    def plot_model_history(self):
        history = self.history

        # http://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        
if __name__ == '__main__':
    dataprocessor = VisionDataProcessor()
    dataprocessor.create_simple_categorical_model()
    #dataprocessor.create_binary_vgg16_model()
    #dataprocessor.create_simple_binary_model()
    #dataprocessor.create_flat_binary_fc_model()
    #dataprocessor.create_doe_model()
    #dataprocessor.create_flat_keras_model()
    #dataprocessor.inception_cross_train()
    dataprocessor.fit_model()
    dataprocessor.plot_model_history()
    #dataprocessor.save_model_to_file()
