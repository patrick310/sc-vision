class Camera(object):
    def __init__(self, name, camType=None):
        self.name = name
		self.camType = camType if camType is not None else "default"
		self.ISO = ISO if ISO is not None else "800"
	
	def getName(self):
        return self.name

    def setCamType(self):												#option 2										
        self.camType = camType
	
	def getCamType(self):
        return self.camType

    def capture(self):
        return "Click!"
		
	def setISO(self):
		self.ISO = 1000

    def __str__(self):
        return "%s is a %s" % (self.name, self.camType)


class RpiCamera(Camera):
    def __init__(self, name):
        self.name = name
        self.camType = "RpiCamera"
	
	def setExposureMode(self, exposure_mode):
		exposure_options = [off, auto, night, nightpreview, backlights, spotlight, sports, snow, beach, verylong, fixefps, antishake, fireworks]
		self.exposure_mode = exposure_mode if exposure_mode is in exposure_options else "auto"
		

class TestCamera(Camera):
    # Simulates a camera and returns a test picture when TestCamera.capture is called
    def __init__(self, name):
        self.name = name
        self.camType = "TestCamera"

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