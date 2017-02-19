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
