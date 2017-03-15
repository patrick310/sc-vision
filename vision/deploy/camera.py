from utils import between_inclusive
from io import BytesIO
from time import sleep
from time import datetime
from datetime import datetime
#from picamera import PiCamera #TODO: check if raspberry pi. Import if yes
from PIL import Image

class Camera(object):
    def __init__(self):
        return None
    
    def get_name(self):    
        return self.name
                            
    def set_name(self, name=None):
        self.name = name if name is not None else "new camera"

    def get_camera_type(self):
        return self.camera_type
    
    def set_camera_type(self, camera_type=None):
        self.camera_type = camera_type if camera_type is not None else "default type"
        
    def get_contrast(self):
        return self.contrast
    
    def set_contrast(self, contrast=None):
        self.contrast = contrast if contrast is not None else str(0)
        #-100 to 100

    def get_brightness(self):
        return self.brightness
    
    def set_brightness(self, brightness=None):
        self.brightness = brightness if brightness is not None else str(50)
        #0 to 100
        
    def get_saturation(self):
        return self.saturation
        
    def set_saturation(self, saturation=None):
        self.saturation = saturation if saturation is not None else str(0)
        #-100 to 100
        
    def get_iso(self):
        return self.iso
    
    def set_iso(self, iso=None):
        self.iso = iso if iso is not None else str(450)
        #100 to 800

    def capture(self):
        return "Click!" 

    def __str__(self):
        return '%s is a %s' % (self.name, self.camType)

    def capture_to_file(self,image):
        count = 1
        image.save('image%s.jpg' % count)
        count += 1
        return True

    def get_next_filePath(output_folder):
        highest_num = 0
        for f in os.listdir(output_folder):
            if os.path.isfile(os.path.join(output_folder, f)):
                file_name = os.path.splitext(f)[0]
                try:
                    file_num = int(file_name)
                    if file_num > highest_num:
                        highest_num = file_num
                except ValueError:
                    'The file name "%s" is not an integer. Skipping' % file_name

        output_file = os.path.join(output_folder, str(highest_num + 1))
        return output_file

    def save_to_file(image, classification):
        # needs to place pictures into appropriate folders based on their known classifications

        #get current working directory
        cwd = os.getcwd()
        #get datestr and change image name Note: wrong but Im not sure how to do it.... yet
        datestr = datetime.strftime(datetime.today(), "%Hh %Mm %Ss %A, %B %Y")
        image= datestr +'.jpg'
        #build new folder's path
        newFolderPath = cwd + "/" + classification
        #create new folder if it doesn't exist already
        if not os.path.exists(newFolderPath):
            os.makedirs(newFolderPath)
        else:
            print(newFolderPath + "already exists!")
        #build path to image in cwd
        imageFilePath = cwd + "/" + image + ".jpg"
        #ensure image file exists
        if os.path.exists(imageFilePath):
            #if so move into new folder
            os.rename(imageFilePath, newFolderPath + "/" + image+ ".jpg")
        else:
            print ("couldn't find" + imageFilePath)
        

class RpiCamera(Camera):

    def __init__(self):
        self.camType = 'RpiCamera'

    def set_exposure_mode(self, exposure_mode):
        exposure_options = [
            off,
            auto,
            night,
            nightpreview,
            backlights,
            spotlight,
            sports,
            snow,
            beach,
            verylong,
            fixefps,
            antishake,
            fireworks,
            ]
        self.exposure_mode = (exposure_mode if exposure_mode
                              in exposure_options else 'auto')

    def set_white_balance(self, awb_mode):
        awb_options = [
            off,
            auto,
            sun,
            cloudshade,
            tungsten,
            fluorescent,
            incadescent,
            flash,
            horizon,
            ]
        self.awb_mode = (awb_mode if awb_mode in awb_options else 'auto'
                         )

    def set_image_effect(self, effect_mode):
        effect_options = [
            none,
            negative,
            solarise,
            whiteboard,
            blackboard,
            sketch,
            denoise,
            emboss,
            oilpaint,
            hatch,
            gpen,
            pastel,
            watercolour,
            film,
            blur,
            saturation,
            colourswap,
            washedout,
            posterie,
            colourpoint,
            colourbalance,
            cartoon,
            ]
        self.effect_mode = (effect_mode if effect_mode
                            in effect_options else 'none')

    def set_sharpness(self, sharpness_mode):
        self.shaprness_mode = (sharpness_mode if sharpness_mode
                               is between_inclusive(-100, 100,
                               sharpness_mode) else str(0))
							
							
    def capture(self):
        # Create the in-memory stream
        stream = BytesIO() 
        with PiCamera() as camera:
            camera.start_preview()
            sleep(2)
            camera.capture(stream, format='jpeg')
            # "Rewind" the stream to the beginning so we can read its content
            stream.seek(0)
            image = Image.open(stream)
        return image
            
class TestCamera(Camera):    
    # Simulates a camera and returns a test picture when TestCamera.capture is called

    def __init__(self, name):
        self.name = name
        self.camType = 'TestCamera'

    def create_test_image():
        from io import BytesIO
        from PIL import Image
        file = BytesIO()
        image = Image.new('RGBA', size=(50, 50), color=(155, 0, 0))
        image.save(file, 'png')
        file.name = 'test.png'
        file.seek(0)
        return image

    def capture(self):
        return TestCamera.create_test_image()
