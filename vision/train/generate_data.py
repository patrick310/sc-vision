from keras.preprocessing.image import ImageDataGenerator
import h5py
import configs

test_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = configs.shear_range,
    zoom_range = configs.zoom_range,
    zca_whitening = configs.zca_whitening,
    rotation_range = configs.rotation_range,
    width_shift_range = configs.width_shift_range,
    height_shift_range = configs.height_shift_range,
    vertical_flip = configs.vertical_flip,
    horizontal_flip = configs.horizontal_flip,
)
#[TODO] Combine test and val flows... refer to scikit test/val split todo in train_network
test_gen = test_datagen.flow_from_directory(
    directory = configs.test_dir,
    target_size = (configs.img_width, configs.img_height),
    batch_size = configs.batch_size,
    color_mode = configs.color_mode,
    class_mode = configs.class_mode
)

val_datagen = ImageDataGenerator(
    rescale = 1./255
)

val_gen = val_datagen.flow_from_directory(
    directory = configs.val_dir,
    target_size = (configs.img_width, configs.img_height),
    batch_size = configs.batch_size,
    color_mode = configs.color_mode,
    class_mode = configs.class_mode
)

print("Creating files.")

test_file = h5py.File(configs.test_fname, "w")
test_image_data = test_file.create_dataset(
    "image_data",
    (configs.nb_test_images, configs.img_width, configs.img_height),
    dtype = "float32"
)
test_class_data = test_file.create_dataset(
    "class_data",
    (configs.nb_test_images, configs.nb_classes),
    dtype = "float32"
)

val_file = h5py.File(configs.val_fname, "w")
val_image_data = val_file.create_dataset(
    "image_data",
    (configs.nb_val_images, configs.img_width, configs.img_height),
    dtype = 'float32'
)
val_class_data = val_file.create_dataset(
    "class_data",
    (configs.nb_val_images, configs.nb_classes),
    dtype = "float32"
)

print("Generating training images.")

counter = 0
while counter != configs.nb_test_images:
    data_pack = test_gen.next() # tuple
    image_data = data_pack[0]
    class_data = data_pack[1]

    for index in range(len(image_data)):
        test_image_data[counter] = image_data[index]
        test_class_data[counter] = class_data[index]
        counter += 1
        if counter == configs.nb_test_images:
            break
    import sys
    sys.stdout.write("  " + str(counter) + "/" + str(configs.nb_test_images) + "\r")
    sys.stdout.flush()
test_file.close()

print("Generating test images.")

counter = 0
while counter != configs.nb_val_images:
    data_pack = val_gen.next()
    image_data = data_pack[0]
    class_data = data_pack[1]

    for index in range(len(image_data)):
        val_image_data[counter] = image_data[index]
        val_class_data[counter] = class_data[index]
        counter += 1
        if counter == configs.nb_val_images:
            break
    sys.stdout.write("  " + str(counter) + "/" + str(configs.nb_val_images) + "\r")
    sys.stdout.flush()
val_file.close()

print("Done.")
