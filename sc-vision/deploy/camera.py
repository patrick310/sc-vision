class Camera(object):
    def __init__(self):
		return True
	
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
			