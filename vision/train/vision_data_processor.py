from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.layers import Activation, Dropout, Flatten, Dense
import h5py
import configs
import numpy as np
import pydot
from keras.utils import plot_model

class VisionDataProcessor():

    def __init__(self):
        self.data_generator = ImageDataGenerator(
        shear_range = configs.shear_range,
        zoom_range = configs.zoom_range,
        zca_whitening = configs.zca_whitening,
        rotation_range = configs.rotation_range,
        width_shift_range = configs.width_shift_range,
        height_shift_range = configs.height_shift_range,
        vertical_flip = configs.vertical_flip,
        horizontal_flip = configs.horizontal_flip,
        )
        
        self.train_generator = self.create_data_generator_from_directory(configs.test_dir)
        self.validation_generator = self.create_data_generator_from_directory(configs.val_dir)
    
    def create_data_generator_from_directory(self, directory):
        generated_generator = self.data_generator.flow_from_directory(
        directory = directory,
        target_size = (configs.img_width, configs.img_height),
        batch_size = configs.batch_size,
        color_mode = configs.color_mode,
        class_mode = configs.class_mode,
        save_to_dir = 'visualize'
        )
        
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
        test_file.close()
        
    def create_simple_keras_model(self):
        input_shape = (1, configs.img_width, configs.img_height)
        
        model = Sequential()
         
        model.add(Conv2D(32, (3, 3),
            padding='same',
            data_format='channels_first',
            input_shape=input_shape))
            # now: model.output_shape == (None, 64, 32, 32)
        
        model.add(Activation('relu'))
        
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Flatten())
        model.add(Dense(75))
        model.add(Dense(40))
        model.add(Dense(10))
        
        model.add(Activation('relu'))
        
        model.add(Dropout(0.75))
        
        model.add(Dense(configs.nb_classes))
        model.add(Activation('sigmoid'))
            
        if configs.print_summary:
            model.summary()
        
        model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
              
        self.model = model
        plot_model(model,to_file='model.png')

    def fit_simple_keras_model(self):
        self.model.fit_generator(
            self.train_generator,
            steps_per_epoch=1,
            epochs=configs.nb_epoch,
            validation_data=self.validation_generator,
            validation_steps=1)
        
if __name__ == '__main__':
    dataprocessor = VisionDataProcessor()
    dataprocessor.create_simple_keras_model()
    dataprocessor.fit_simple_keras_model()
    print("Done.")
