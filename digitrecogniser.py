# -*- coding: utf-8 -*-

import sys
from PyQt4 import QtGui, QtCore

class Point:
    x = 0
    y = 0
    def __init__(self):
        self.x = 0
        self.y = 0
    def __init__(self, _x, _y):
        self.x = _x
        self.y = _y
    def set(self, _x, _y):
        self.x = _x
        self.y = _y

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

class Painter(QtGui.QWidget):
    ptArray = PointArray()
    mouseLocation = Point(0, 0)
    lastPos = Point(0, 0)
    isPainting = False

    def __init__(self, parent):
        super(Painter, self).__init__()
        mouseLocation = Point(0, 0)
        lastPos = Point(0, 0)
        isPainting = False
    def mousePressEvent(self, event):
        isPainting = True

    def mouseMoveEvent(self, event):
        if(self.isPainting == True):
            self.mouseLocation = Point(event.x(), event.y())
            if(self.lastPos.x != self.mouseLocation.x and self.lastPos.y != self.mouseLocation.y):
                self.lastPos = Point(event.x(), event.y())
                ptArray.addPoint(event.x(), event.y())

    def mouseReleaseEvent(self, event):
        self.isPainting = False

    def paintEvent(self,event):
        painter = QtGui.QPainter()
        painter.begin(self)
        self.drawLines(event, painter)
        painter.end()

class MainWindow(QtGui.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.initUI()

    def initUI(self):
        cWidget = Painter(QtGui.QWidget)
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


def main():
    app = QtGui.QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()