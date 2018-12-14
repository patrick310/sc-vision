# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'slidebar_pop.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_PopWindow(object):
    def setupUi(self, PopWindow):
        PopWindow.setObjectName("PopWindow")
        PopWindow.resize(519, 112)
        self.centralwidget = QtWidgets.QWidget(PopWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(0, 0, 521, 81))
        self.widget.setObjectName("widget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)
        self.label_2 = QtWidgets.QLabel(self.widget)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_2.addWidget(self.label_2)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.f_slider = QtWidgets.QSlider(self.widget)
        self.f_slider.setMaximum(255)
        self.f_slider.setOrientation(QtCore.Qt.Horizontal)
        self.f_slider.setObjectName("f_slider")
        self.verticalLayout.addWidget(self.f_slider)
        self.e_slider = QtWidgets.QSlider(self.widget)
        self.e_slider.setMaximum(12)
        self.e_slider.setOrientation(QtCore.Qt.Horizontal)
        self.e_slider.setObjectName("e_slider")
        self.verticalLayout.addWidget(self.e_slider)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.horizontalLayout_2.addLayout(self.horizontalLayout)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.fpop_value = QtWidgets.QLabel(self.widget)
        self.fpop_value.setMinimumSize(QtCore.QSize(40, 0))
        self.fpop_value.setAlignment(QtCore.Qt.AlignCenter)
        self.fpop_value.setObjectName("fpop_value")
        self.verticalLayout_3.addWidget(self.fpop_value)
        self.epop_value = QtWidgets.QLabel(self.widget)
        self.epop_value.setMinimumSize(QtCore.QSize(40, 0))
        self.epop_value.setAlignment(QtCore.Qt.AlignCenter)
        self.epop_value.setObjectName("epop_value")
        self.verticalLayout_3.addWidget(self.epop_value)
        self.horizontalLayout_2.addLayout(self.verticalLayout_3)
        PopWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(PopWindow)
        self.statusbar.setObjectName("statusbar")
        PopWindow.setStatusBar(self.statusbar)

        self.retranslateUi(PopWindow)
        QtCore.QMetaObject.connectSlotsByName(PopWindow)

    def retranslateUi(self, PopWindow):
        _translate = QtCore.QCoreApplication.translate
        PopWindow.setWindowTitle(_translate("PopWindow", "MainWindow"))
        self.label.setText(_translate("PopWindow", "Focus"))
        self.label_2.setText(_translate("PopWindow", "Exposure"))
        self.fpop_value.setText(_translate("PopWindow", "0"))
        self.epop_value.setText(_translate("PopWindow", "0"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    PopWindow = QtWidgets.QMainWindow()
    ui = Ui_PopWindow()
    ui.setupUi(PopWindow)
    PopWindow.show()
    sys.exit(app.exec_())

