class Camera(object):

    def __init__(self, name, camType=None):
        self.name = name
        self.camType = (camType if camType is not None else 'default')
        self.ISO = (ISO if ISO is not None else '800')

    def getName(self):
        return self.name

    def setCamType(self): 
        self.camType = camType

    def getCamType(self):
        return self.camType

    def capture(self):
        return 'Click!'

    def setISO(self):
        self.ISO = 1000

    def __str__(self):
        return '%s is a %s' % (self.name, self.camType)


class RpiCamera(Camera):

    def __init__(self, name):
        self.name = name
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

    def between_incluside(low, high, value):
        if value >= low and value <= high:
            return True
        else:
            return False


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
        return file

    def capture(self):
        return TestCamera.create_test_image()



			
