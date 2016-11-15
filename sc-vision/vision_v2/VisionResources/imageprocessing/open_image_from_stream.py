import PIL
from PIL import Image
import io

def open_image_from_stream(stream):
    # Make sure to rewind stream to beginning of data.
    stream.seek(0)
    return Image.open(stream)
