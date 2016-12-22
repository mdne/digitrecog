# -*- coding: utf-8 -*-

import sys
from PyQt4 import QtGui, QtCore
from math import sqrt

'''
TODO разобраться с атрибутами листа и дописать редьюс
'''
class Reducer:
    #def __init__(self):

    def distance(self, a, b):
        return  sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    def point_line_distance(self, point, start, end):
        if (start == end):
            return self.distance(point, start)
        else:
            n = abs((end.x - start.x) * (start.y - point.y) - (start.x - point.x) * (end.y - start.y))
            d = self.distance(end, start)
            return n / d

    def rdp(self, points, epsilon):
        dmax = 0.0
        index = 0
        for i in range(1, len(points) - 1):
            d = self.point_line_distance(points[i], points[0], points[-1])
            if d > dmax:
                index = i
                dmax = d
        if dmax >= epsilon:
            results = self.rdp(points[:index+1], epsilon)[:-1] + self.rdp(points[index:], epsilon)
        else:
            results = [points[0], points[-1]]
        return results


class Point:
    x = 0
    y = 0
    def __init__(self):
        self.x = 0
        self.y = 0
    def __init__(self, _x, _y):
        self.x = _x
        self.y = _y

class PointList:
    ptList = []
    def __init__(self):
        self.ptList = []
    def __getitem__(self, item):
        return self.ptList[item]
    def __len__(self):
        return len(self.ptList)
    def getSize(self):
        return len(self.ptList)
    def addPoint(self, _x, _y):
        pt = Point(_x, _y)
        self.ptList.append(pt)
    def getPoint(self, idx):
        return self.ptList[idx]
    def removePoints(self):
        del self.ptList[:]


class MainWidget(QtGui.QWidget):
    reducer = 0
    ptList = 0
    mouseLocation = Point(0, 0)
    lastPos = Point(0, 0)
    isPainting = False
    isPainted = False
    clearButton = 0
    recogButton = 0

    def __init__(self):
        super(MainWidget, self).__init__()
        self.reducer = Reducer()
        self.ptList = PointList()
        self.mouseLocation = Point(0, 0)
        self.lastPos = Point(0, 0)
        self.isPainting = False
        self.isPainted = False
        self.clearButton = 0
        self.recogButton = 0
        self.initUI()

    def initUI(self):
        self.clearButton = QtGui.QPushButton("Clear")
        QtCore.QObject.connect(self.clearButton, QtCore.SIGNAL("clicked()"),self.clearArea)
        self.recogButton = QtGui.QPushButton("Recognize")
        QtCore.QObject.connect(self.recogButton, QtCore.SIGNAL("clicked()"),self.reducePt)

        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(self.clearButton)
        hbox.addWidget(self.recogButton)
        vbox = QtGui.QVBoxLayout()
        vbox.addStretch(1)
        vbox.addLayout(hbox)
        self.setLayout(vbox)

        self.setGeometry(300, 300, 500, 500)
        self.show()

    def mousePressEvent(self, event):
        self.isPainting = True

    def mouseMoveEvent(self, event):
        if(self.isPainting == True):
            if(self.isPainted):
                self.clearArea()
                self.isPainted = False

            self.mouseLocation = Point(event.x(), event.y())
            if(self.lastPos.x != self.mouseLocation.x and self.lastPos.y != self.mouseLocation.y):
                self.lastPos = Point(event.x(), event.y())
                self.ptList.addPoint(event.x(), event.y())
                self.repaint()

    def mouseReleaseEvent(self, event):
        self.isPainting = False
        self.isPainted = True

    def paintEvent(self,event):
        painter = QtGui.QPainter()
        painter.begin(self)
        self.drawLines(event, painter)
        painter.end()

    def drawLines(self, event, painter):
        painter.setRenderHint(QtGui.QPainter.Antialiasing);
        for i in range(len(self.ptList) - 1):
            pt1 = self.ptList[i]
            pt2 = self.ptList[i+1]
            pen = QtGui.QPen(QtGui.QColor(255,0,0), 4, QtCore.Qt.SolidLine)
            painter.setPen(pen)
            painter.drawLine(pt1.x,pt1.y,pt2.x,pt2.y)

    def clearArea(self):
        self.ptList.removePoints()
        self.repaint()

    def reducePt(self):
        print self.ptList.getSize()
        array = self.reducer.rdp(self.ptList, 10.0)
        print len(array)
        self.clearArea()

        self.ptList = array
        self.repaint()


def main():
    app = QtGui.QApplication(sys.argv)
    ex = MainWidget()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()