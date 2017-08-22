import os
import os.path
import io



def create_boolean_image_lists(image_dir):
    for file in generate_file_list(image_dir):
        with open("bolt.txt", "a") as myfile:
            myfile.write(str(file) + " -1\n")

def generate_file_list(image_dir, filter=".jpg"):
    data = []
    for dirpath, dirnames, filenames in os.walk(image_dir):
        for filename in [f for f in filenames if f.endswith(filter)]:
            data.append(filename)

    return data


if __name__ == '__main__':
    create_boolean_image_lists('C://Users//patri//PycharmProjects//sc-vision//tests//data//neg')