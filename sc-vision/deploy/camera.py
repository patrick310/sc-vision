class Camera(object):
    def __init__(self, name, camType):
        self.name = name
        self.camType = camType

    def getName(self):
        return self.name

    def getCamType(self):
        return self.camType

    def capture(self):
        return "Click!"

    def __str__(self):
        return "%s is a %s" % (self.name, self.camType)


class RpiCamera(Camera):
    def __init__(self, name):
        self.name = name
        self.camType = "RpiCamera"


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
