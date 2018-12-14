import cv2
import PIL.Image, PIL.ImageTk
import pyzbar.pyzbar as pyzbar
import numpy as np
import PyQt5
from PyQt5 import QtGui, QtWidgets, QtTest
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QTimer, QDateTime, QTime
from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow
from PyQt5.uic import loadUi
import sys
import logging
import time
# from testreal import Deployment, VideoCapture
from vision_gui import Ui_MainWindow
import os
import filter_file
from filter_file import *
import inspect
import imp
import fileinput
import string



logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()
logger.handlers = [] # This is the key thing for the question!
# Start defining and assigning your handlers here
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s: %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class GuiLogger(logging.Handler):
    def emit(self, record):
        App.log = logger.getEffectiveLevel()
        self.edit.appendPlainText(self.format(record)) # implementation of append_line omitted



class App(QMainWindow):
    def __init__(self):
        super(App, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.timer = QTimer(self)
        self.button_timer = QTimer(self)
        self.logging_levels = ["DEBUG", "INFO", "WARN", "ERROR", "FATAL"]
        self.logger_fill()
        self.blank_image = np.zeros((20, 20, 3), np.uint8)
        self.win1 = self.blank_image
        self.win2 = self.blank_image
        self.win3 = self.blank_image
        self.win4 = self.blank_image
        self.displayImage(self.win1, 1)
        self.displayImage(self.win2, 2)
        self.displayImage(self.win3, 3)
        self.displayImage(self.win4, 4)
        # self.filter_list = vdir(filter_file)
        # print(self.filter_list)
        self.filter_run = []
        self.filter_list = []
        self.add_to_list()
        self.filter_fill()
        self.filter_list.clear()
        self.deployment = None
        self.video_source = 0
        self.ui.video_input.addItem('0')
        self.ui.video_input.addItem('1')
        self.ui.video_input.addItem('2')
        self.ui.video_input.activated.connect(self.video_input_set)
        self.height = 2048
        self.width = 2048
        self.auto_focus = True
        self.focus = None
        self.auto_exposure = True
        self.exposure = None
        self.ui.s_expose_auto_button.setChecked(True)
        self.ui.s_focus_auto_button.setChecked(True)
        self.file_path = "C:\\Users\\MBUSI\\Desktop\\Vision_Projects\\Ratio\\share\\"
        self.start_clock()
        self.ui.start_button.clicked.connect(self.start_camera)
        self.ui.stop_button.clicked.connect(self.stop_camera)
        self.ui.screen_shot.clicked.connect(self.screen_shot)
        self.ui.access_directory_button.clicked.connect(self.access_directory)
        self.ui.carimg1.mousePressEvent = self.show_carimg1
        self.ui.carimg2.mousePressEvent = self.show_carimg2
        self.ui.carimg3.mousePressEvent = self.show_carimg3
        self.ui.carimg4.mousePressEvent = self.show_carimg4
        self.ui.s_focus_auto_button.clicked.connect(self.set_autofocus)
        self.ui.s_focus_auto_button.clicked.connect(self.start_camera)
        self.ui.s_focus_default_button.clicked.connect(self.set_defaultfocus)
        self.ui.s_focus_default_button.clicked.connect(self.start_camera)
        self.ui.s_expose_auto_button.clicked.connect(self.set_autoexposure)
        self.ui.s_expose_auto_button.clicked.connect(self.start_camera)
        self.ui.s_expose_default_button.clicked.connect(self.set_defaultexposure)
        self.ui.s_expose_default_button.clicked.connect(self.start_camera)
        self.ui.s_focus_slide_button.clicked.connect(self.ui.slide)
        self.ui.ui.f_slider.sliderMoved.connect(self.set_slidefocus)
        self.ui.ui.f_slider.sliderReleased.connect(self.start_camera)
        self.ui.s_expose_slide_button.clicked.connect(self.ui.slide)
        self.ui.ui.e_slider.sliderMoved.connect(self.set_slideexposure)
        self.ui.ui.e_slider.sliderReleased.connect(self.start_camera)
        self.ui.filter_options.activated.connect(self.edit_run_list)
        self.ui.filter_options.activated.connect(self.update_listW)
        self.ui.add_filter_settings.clicked.connect(self.ui.code)
        self.ui.uic.code_submit.clicked.connect(self.update_code)
        self.ui.comboBox.activated.connect(self.logger_set)
        h = GuiLogger()
        h.edit = self.ui.log_out
        logging.getLogger().addHandler(h)


    def __del__(self):
        if self.deployment.video.cap.isOpened():
            self.deployment.video.cap.release()

    def logger_fill (self):
        for level in self.logging_levels:
            self.ui.comboBox.addItem(level)
        logging.getLogger().setLevel(logging.NOTSET)

    def logger_set(self):
        logger.debug((str(self.ui.comboBox.currentText()) + "=" + str(self.logging_levels[1])))
        if (str(self.ui.comboBox.currentText()) == str(self.logging_levels[0])):
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("DEBUG is Set")
            self.ui.log_out.setStyleSheet("background-color: rgb(85, 0, 127); color: rgb(255, 255, 255);")
        elif (str(self.ui.comboBox.currentText()) == str(self.logging_levels[1])):
            logging.getLogger().setLevel(logging.INFO)
            logger.info("INFO is Set")
            self.ui.log_out.setStyleSheet("background-color: rgb(0, 0, 127); color: rgb(255, 255, 255);")
        elif (str(self.ui.comboBox.currentText()) == str(self.logging_levels[2])):
            logging.getLogger().setLevel(logging.WARN)
            logger.warn("WARN is Set")
            self.ui.log_out.setStyleSheet("background-color: rgb(255, 85, 0); color: rgb(255, 255, 255);")
        elif (str(self.ui.comboBox.currentText()) == str(self.logging_levels[3])):
            logging.getLogger().setLevel(logging.ERROR)
            logger.error("ERROR is Set")
            self.ui.log_out.setStyleSheet("background-color: rgb(180, 0, 0); color: rgb(255, 255, 255);")
        elif (str(self.ui.comboBox.currentText()) == str(self.logging_levels[4])):
            logging.getLogger().setLevel(logging.FATAL)
            logger.fatal("FATAL is Set")
            self.ui.log_out.setStyleSheet("background-color: rgb(0, 0, 0); color: rgb(255, 255, 255);")


    def reindent(self, s):
        s = s.split('\n')
        s = [('\t ') + line.lstrip() for line in s]
        s = '\n'.join(s)
        return s

    def update_code(self):
        filter_name = self.ui.uic.code_title.text()
        filter_body = self.ui.uic.code_body.toPlainText()
        filter_body = self.reindent(filter_body)
        f_end_string = "#f_end_string"
        d_end_string = "#d_end_string"
        function_add = "def " + filter_name + "(image, value=None): \n" + "\timage = np.asarray(image)\n" + \
                       filter_body + "\n" + "\treturn[image, value]" + "\n" + f_end_string
        dict_add = "\t\t\t\t" + r'"' + filter_name + r'"' + ' : ' + filter_name + "," + "\n" + d_end_string
        with fileinput.FileInput("filter_file.py", inplace=True, backup='.bak') as file:
            for line in file:
                print(line.replace(f_end_string, function_add), end='')
        with fileinput.FileInput("filter_file.py", inplace=True, backup='.bak') as file:
            for line in file:
                print(line.replace(d_end_string, dict_add), end='')
        imp.reload(filter_file)
        self.ui.filter_options.clear()
        self.add_to_list()
        self.filter_fill()
        self.filter_list.clear()
        logger.info("update code accesed")


    def update_listW(self):
        self.ui.listWidget.clear()
        for item in self.filter_list:
            self.ui.listWidget.addItem(item)


    def edit_run_list(self):
        temp = self.ui.filter_options.currentText()
        print(temp)
        if temp not in self.filter_list:
            self.filter_run.append(filter_file.filter_dict[temp])
            self.filter_list.append(temp)
        else:
            self.filter_run.remove(filter_file.filter_dict[temp])
            self.filter_list.remove(temp)



    def add_to_list(self):
        for key, value in filter_file.filter_dict.items():
            self.filter_list.append(key)


    def filter_fill(self):
        for self.filter in self.filter_list:
            self.ui.filter_options.addItem(self.filter)
        logger.info("filer file accesed")
        logger.info(filter_file.filter_dict)



    def set_slideexposure(self):
        self.auto_exposure = False
        self.exposure = self.ui.ui.e_slider.value() * -1
        self.ui.ui.epop_value.setText(str(self.exposure))
        logger.info(self.exposure)


    def set_slidefocus(self):
        self.auto_focus = False
        self.focus = self.ui.ui.f_slider.value()
        self.ui.ui.fpop_value.setText(str(self.focus))
        logger.info(self.focus)


    def set_defaultexposure(self):
        self.auto_exposure = False
        self.exposure = -7
        logger.debug("self.focus and self.auto_focus changed")
    def set_autoexposure(self):
        self.auto_exposure = True
        self.exposure = None
        logger.debug("self.exposure and self.auto_exposure changed")

    def video_input_set(self):
        self.video_source = int(self.ui.video_input.currentText())

    def set_defaultfocus(self):
        self.auto_focus = False
        self.focus = 27
        logger.debug("self.focus and self.auto_focus changed")
    def set_autofocus(self):
        self.auto_focus = True
        self.focus = None
        logger.debug("self.focus and self.auto_focus changed")

    def show_carimg1(self, event):
        cv2.imshow("Preview1", self.win1)

    def show_carimg2(self, event):
        cv2.imshow("Preview2", self.win2)

    def show_carimg3(self, event):
        cv2.imshow("Preview3", self.win3)

    def show_carimg4(self, event):
        cv2.imshow("Preview4", self.win4)

    def access_directory(self):
        os.startfile(self.file_path)

    def screen_shot(self):
        frame = self.deployment.process_frame()
        self.file_name = 'img-%s.jpg' % time.strftime("%Y-%m-%d-%H-%M-%S")
        self.file_path = "C:\\Users\\MBUSI\\Desktop\\Vision_Projects\\Ratio\\share\\"
        self.file_save = self.file_path + self.file_name
        logging.debug(self.file_save)
        self.save(self.file_save, frame[0])


    def start_clock(self):
        self.button_timer.setInterval(1000)
        self.button_timer.timeout.connect(self.display_time)
        self.button_timer.start()


    def display_time(self):
        self.ui.time_label.setText(QDateTime.currentDateTime().toString())



    def test(self):
        logging.debug('damn, a bug')
        logging.info('something to remember')
        logging.warning('that\'s not right')
        logging.error('foobar')


    def stop_camera(self):
        self.timer.stop()
        # self.__del__()
        QtTest.QTest.qWait(100)
        if self.deployment is not None:
            self.deployment.video.cap.release()
            self.deployment = None


    def start_camera(self):
        self.stop_camera()
        self.deployment = Deployment(filters=self.filter_run,
                                     video=VideoCapture(video_source=self.video_source, width=self.width,
                                                        height=self.height, auto_focus=self.auto_focus, focus=self.focus,
                                                        exposure=self.exposure, auto_exposure=self.auto_exposure))
        logger.debug("Camera Deployed")
        self.timer.setInterval(10)
        self.timer.timeout.connect(self.update)
        self.timer.start()


    def save(self, file_save=None, img=None):
        # self.deployment.save_img(file_save=file_save, image=img)
        self.win4 = self.win3
        self.win3 = self.win2
        self.win2 = self.win1
        self.win1 = img
        self.displayImage(self.win1, 1)
        self.displayImage(self.win2, 2)
        self.displayImage(self.win3, 3)
        self.displayImage(self.win4, 4)

    def update(self):
        logger.debug("update")
        frame = self.deployment.process_frame()
        self.displayImage(frame[0], 0)

    def displayImage(self, frame, window=0):
        qformat=QImage.Format_Indexed8
        if len(frame.shape)==3 :
            if frame.shape[2]==4 :
                qformat=QImage.Format_RGBA8888
            else:
                    qformat=QImage.Format_RGB888
        if window == 0:
            outImage=QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0],qformat)
            outImage=outImage.rgbSwapped()

            self.ui.videoout.setPixmap(QtGui.QPixmap.fromImage(outImage))
            self.ui.videoout.setScaledContents(True)
        elif window == 1:
            outImage = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], qformat)
            outImage = outImage.rgbSwapped()
            self.ui.carimg1.setPixmap(QtGui.QPixmap.fromImage(outImage))
            self.ui.carimg1.setScaledContents(True)
        elif window == 2:
            outImage = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], qformat)
            outImage = outImage.rgbSwapped()
            self.ui.carimg2.setPixmap(QtGui.QPixmap.fromImage(outImage))
            self.ui.carimg2.setScaledContents(True)
        elif window == 3:
            outImage = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], qformat)
            outImage = outImage.rgbSwapped()
            self.ui.carimg3.setPixmap(QtGui.QPixmap.fromImage(outImage))
            self.ui.carimg3.setScaledContents(True)
        elif window == 4:
            outImage = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], qformat)
            outImage = outImage.rgbSwapped()
            self.ui.carimg4.setPixmap(QtGui.QPixmap.fromImage(outImage))
            self.ui.carimg4.setScaledContents(True)



class VideoCapture:

    def __init__(self, video_source=None, width=None, height=None, auto_focus=None,
                 focus=None, exposure=None, auto_exposure=None):
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            raise ValueError("Unable to open video source", video_source)
        if width is not None:
            self.test = self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if self.test:
                logger.info("Sucessfully Returned width")
            logger.info(width)
        else:
            logger.warning('Width Fail')
        if height is not None:
            self.test = self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if self.test:
                logger.info("Sucessfully Returned height")
            else:
                logger.warning('Height Fail')
            logger.info(height)
        if auto_focus is not None:
            self.test = self.cap.set(cv2.CAP_PROP_AUTOFOCUS, auto_focus)
            if self.test:
                logger.info("Sucessfully Returned auto_focus")
            else:
                logger.warning('Auto Fail')
            logger.info(auto_focus)
        if focus is not None:
            self.test = self.cap.set(cv2.CAP_PROP_FOCUS, focus)
            if self.test:
                logger.info("Sucessfully Returned focus")
            else:
                logger.warning('Focus Fail')
            logger.info(focus)
        if auto_exposure is not None:
            self.test = self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_exposure)
            if self.test:
                logger.info("Sucessfully Returned auto_exposure")
            else:
                logger.warning('AutoExpo Fail')
            logger.info(auto_exposure)
        if exposure is not None:
            self.test = self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
            if self.test:
                logger.info("Sucessfully Returned exposure")
            else:
                logger.warning('exposure Fail')
            logger.info(exposure)


        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        if self.width == width:
            logger.debug("width changed sucessful")
        else:
            logger.warning('Width Fail')
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if self.height == height:
            logger.debug("height changed sucessful")
        else:
            logger.warning('height Fail')
        self.auto_focus = self.cap.get(cv2.CAP_PROP_AUTOFOCUS)
        if self.auto_focus == auto_focus:
            logger.debug("auto changed sucessful")
        else:
            logger.warning('auto Fail')
        self.focus = self.cap.get(cv2.CAP_PROP_FOCUS)
        if self.focus == focus:
            logger.debug("focus changed sucessful")
        else:
            logger.warning('focus Fail')

        self.auto_exposure = self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        if self.auto_exposure == auto_exposure:
            logger.debug("autoexpsoe changed sucessful")
        else:
            logger.warning('auto expsoure Fail')
        self.exposure = self.cap.get(cv2.CAP_PROP_EXPOSURE)
        if self.exposure == exposure:
            logger.debug("exposure changed sucessful")
        else:
            logger.warning('expsoure Fail')
        logger.debug(self.width)
        logger.debug(self.height)
        logger.debug(self.auto_focus)
        logger.debug(self.focus)
        logger.debug(self.auto_exposure)
        logger.debug(self.exposure)


    def get_frame(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return ret, frame
            else:
                return ret, None
        else:
            return False, None


class Deployment:
    '''All filters must take an image (write an assert)
    '''

    def __init__(self, filters=None, video=None):
        if filters is None:
            filters = []
        if video is None:
            video = VideoCapture()

        self.filters = filters
        self.video = video

    def get_frame(self):
        return self.video.get_frame()

    def save_img(self, file_save, image):
        cv2.imwrite(file_save, image)
        return True


    # def update_filters(self, )

    def process_frame(self):
        start_time = time.time()
        ret, frame = self.get_frame()
        frame = [frame, None]
        for method in self.filters:
            frame = method(image=frame[0], value=frame[1])
        if (time.time() - start_time) > 0.001:
            logger.debug('Frame process time was {0:0.1f} seconds'.format(
                time.time() - start_time) + ' for an FPS of {0:0.1f}'.format(
                1 / (time.time() - start_time)))

        return frame


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = App()
    window.setWindowTitle('AutoLens')
    window.show()
    sys.exit(app.exec_())