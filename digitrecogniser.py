# -*- coding: utf-8 -*-

import sys
from PyQt4 import QtGui, QtCore

class MainWindow(QtGui.QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.initUI()

    def initUI(self):
        cWidget = QtGui.QWidget()
        self.setCentralWidget(cWidget)

        clearButton = QtGui.QPushButton("Clear")
        recogButton = QtGui.QPushButton("Recognize")
        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(clearButton)
        hbox.addWidget(recogButton)
        vbox = QtGui.QVBoxLayout()
        vbox.addStretch(1)
        vbox.addLayout(hbox)
        cWidget.setLayout(vbox)

        self.setGeometry(300, 300, 600, 400)
        self.show()

class Point:
    x = 0;
    y = 0;
    def __init__(self):
        self.x = 0;
        self.y = 0;
    def __init__(self, _x, _y):
        self.x = _x;
        self.y = _y;
    def set(self, _x, _y):
        self.x = _x;
        self.y = _y;

class PointArray:
    pArray = []
    def __init__(self):
        self.pArray = []
    def size(self):
        return len(self.pArray)
    def addPoint(self, x, y):
        pt = Point(self, x, y)
        self.pArray.append(pt)
    def getPoint(self, idx):
        return self.pArray[idx]
    def removePoints(self):
        del pArray[:]

def main():
    app = QtGui.QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()