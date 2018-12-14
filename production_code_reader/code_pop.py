# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'code_pop.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Code(object):
    def setupUi(self, Code):
        Code.setObjectName("Code")
        Code.resize(833, 680)
        self.centralwidget = QtWidgets.QWidget(Code)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 831, 651))
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.layoutWidget = QtWidgets.QWidget(self.tab)
        self.layoutWidget.setGeometry(QtCore.QRect(0, 0, 821, 621))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.layoutWidget)
        self.label.setStyleSheet("font: 8pt \"MS Shell Dlg 2\";")
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.code_title = QtWidgets.QLineEdit(self.layoutWidget)
        self.code_title.setObjectName("code_title")
        self.verticalLayout.addWidget(self.code_title)
        self.label_2 = QtWidgets.QLabel(self.layoutWidget)
        self.label_2.setObjectName("label_2")
        self.verticalLayout.addWidget(self.label_2)
        self.code_body = QtWidgets.QPlainTextEdit(self.layoutWidget)
        self.code_body.setObjectName("code_body")
        self.verticalLayout.addWidget(self.code_body)
        self.code_submit = QtWidgets.QPushButton(self.layoutWidget)
        self.code_submit.setObjectName("code_submit")
        self.verticalLayout.addWidget(self.code_submit)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.tableView = QtWidgets.QTableView(self.tab_2)
        self.tableView.setGeometry(QtCore.QRect(0, 0, 831, 621))
        self.tableView.setObjectName("tableView")
        self.tabWidget.addTab(self.tab_2, "")
        Code.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(Code)
        self.statusbar.setObjectName("statusbar")
        Code.setStatusBar(self.statusbar)

        self.retranslateUi(Code)
        self.tabWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(Code)

    def retranslateUi(self, Code):
        _translate = QtCore.QCoreApplication.translate
        Code.setWindowTitle(_translate("Code", "MainWindow"))
        self.label.setText(_translate("Code", "Filter Name:"))
        self.label_2.setText(_translate("Code", "Filter Code:"))
        self.code_submit.setText(_translate("Code", "Sumbit"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("Code", "Code Filter"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("Code", "Build Filter"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Code = QtWidgets.QMainWindow()
    ui = Ui_Code()
    ui.setupUi(Code)
    Code.show()
    sys.exit(app.exec_())

