import RPi.GPIO as GPIO

import JSONEngine
import MessageEngine

je = JSONEngine.JSONEngine("config.json")
configs = je.load()

me = MessageEngine.MessageEngine()

class GPIOActivator:
    ON = 1
    OFF = 2
    def __init__(self, description, pin, message_handler):
        msg = "setting up pin " + str(pin) + " for " + description
        message_handler.handle_message(msg, MessageEngine.INFO_IDENT)

        self.description = description
        self.pin = pin

        GPIO.setup(self.pin, GPIO.OUT)
        self.deactivate()

    def activate(self):
        GPIO.output(self.pin, GPIO.HIGH)
        self.state = self.ON

    def deactivate(self):
        GPIO.output(self.pin, GPIO.LOW)
        self.state = self.OFF

    def is_on(self):
        return self.state == self.ON

class GPIOEngine:
    def __init__(self, pin_list, message_handler):
        self.pins = list()
        self.message_handler = message_handler

        self.message_handler.handle_message("enabling GPIO pins.", MessageEngine.INFO_IDENT)
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        for description, pin in pin_list.iteritems():
            self.pins.append(GPIOActivator(description = description, pin = pin, message_handler = self.message_handler))

        msg = str(len(self.pins)) + " pins assigned"
        self.message_handler.handle_message(msg, MessageEngine.INFO_IDENT)

    def activate(self, description = None, pin = None):
        if description is None and pin is None:
            msg = "must provide pin description or pin number to activate"
            self.message_handler.handle_message(msg, MessageEngine.ERROR_IDENT)

        if description is None:
            for spot in self.pins:
                if spot.pin == pin:
                    spot.activate()
                    msg = "(" + spot.description + ", " + str(pin) + ")" + " activated"
                    self.message_handler.handle_message(msg, MessageEngine.INFO_IDENT)
                    return
        else:
            for spot in self.pins:
                if spot.description == description:
                    spot.activate()
                    msg = "(" + spot.description + ", " + str(pin) + ")" + " activated"
                    self.message_handler.handle_message(msg, MessageEngine.INFO_IDENT)
                    return
        msg = "invalid description or pin"
        self.message_handler.handle_message(msg, MessageEngine.ERROR_IDENT)

    def deactivate(self, description = None, pin = None):
        if description is None and pin is None:
            msg = "must provide pin description or pin number to deactivate"
            self.message_handler.handle_message(msg, MessageEngine.ERROR_IDENT)

        if description is None:
            for spot in self.pins:
                if spot.pin == pin:
                    spot.deactivate()
                    msg = "(" + spot.description + ", " + str(pin) + ")" + " deactivated"
                    self.message_handler.handle_message(msg, MessageEngine.INFO_IDENT)
                    return
        else:
            for spot in self.pins:
                if spot.description == description:
                    spot.deactivate()
                    msg = "(" + spot.description + ", " + str(pin) + ")" + " deactivated"
                    self.message_handler.handle_message(msg, MessageEngine.INFO_IDENT)
                    return
        msg = "invalid description or pin"
        self.message_handler.handle_message(msg, MessageEngine.ERROR_IDENT)

    def is_on(self, description = None, pin = None):
        if description is None and pin is None:
            msg = "must provide pin description or pin number to interface"
            self.message_handler.handle_message(msg, MessageEngine.ERROR_IDENT)

        if description is None:
            for spot in self.pins:
                if spot.pin == pin:
                    return spot.is_on()
        else:
            for spot in self.pins:
                if spot.description == description:
                    return spot.is_on()
        msg = "invalid description or pin"
        self.message_handler.handle_message(msg, MessageEngine.ERROR_IDENT)

ge = GPIOEngine(configs["GPIO"], me)
ge.activate(pin = 19)
ge.deactivate(pin = 19)
ge.activate(pin = 19)
ge.activate(description = "camera")
