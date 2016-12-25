# -*- coding: utf-8 -*-

import sys
from PyQt4 import QtGui, QtCore
from math import sqrt

class Reducer:
    weights = []

    def __init__(self):
        self.weights = []

    def distance(self, a, b):
        return  sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def point_line_distance(self, point, start, end):
        if (start == end):
            return self.distance(point, start)
        else:
            n = abs((end[0] - start[0]) * (start[1] - point[1]) - (start[0] - point[0]) * (end[1] - start[1]))
            d = self.distance(end, start)
            return n / d

    def douglasPeucker(self, points, start, end):
        dmax = 0.0
        index = 0
        for i in range(start+1, end):
            d = self.point_line_distance(points[i], points[start], points[end])
            if d > dmax:
                index = i
                dmax = d
        if(index == 0): return 
        self.weights[index] = dmax
        self.douglasPeucker(points, start, index)
        self.douglasPeucker(points, index, end)

    def rdp(self, points, numPoints):
        self.weights = [0.0] * len(points)
        self.douglasPeucker(points, 0, len(points)-1)

        self.weights[0] = float("inf")
        self.weights[len(points)-1] = float("inf")
        weightsTmp = self.weights[:]
        weightsTmp.sort(reverse = True)
        maxTolerace = weightsTmp[numPoints - 1]
        print maxTolerace
        result = []
        for i in range(0, len(points)):
            if(self.weights[i] >= maxTolerace):
                result.append(points[i])
        return result

    def pointNormalize(self, points):
        return 0

class MainWidget(QtGui.QWidget):
    reducer = 0
    ptList = []
    mouseLocation = [0, 0]
    lastPos = [0, 0]
    isPainting = False
    isPainted = False
    clearButton = 0
    recogButton = 0

    def __init__(self):
        super(MainWidget, self).__init__()
        self.reducer = Reducer()
        self.ptList = []
        self.mouseLocation = [0, 0]
        self.lastPos = [0, 0]
        self.isPainting = False
        self.isPainted = False
        self.clearButton = 0
        self.recogButton = 0
        self.initUI()

    def initUI(self):
        self.clearButton = QtGui.QPushButton("Clear")
        QtCore.QObject.connect(self.clearButton, QtCore.SIGNAL("clicked()"),self.clearArea)
        self.recogButton = QtGui.QPushButton("Recognize")
        #QtCore.QObject.connect(self.recogButton, QtCore.SIGNAL("clicked()"),self.reducePt)

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

            self.mouseLocation = [event.x(), event.y()]
            if(self.lastPos[0] != self.mouseLocation[0] and self.lastPos[1] != self.mouseLocation[1]):
                self.lastPos = [event.x(), event.y()]
                self.ptList.append([event.x(), event.y()])
                self.repaint()

    def mouseReleaseEvent(self, event):
        self.isPainting = False
        self.isPainted = True
        self.reducePt()

    def paintEvent(self,event):
        painter = QtGui.QPainter()
        painter.begin(self)
        self.drawLines(event, painter)
        painter.end()

    def drawLines(self, event, painter):
        painter.setRenderHint(QtGui.QPainter.Antialiasing);
        for i in range(0, len(self.ptList)-1):
            pt1 = self.ptList[i]
            pt2 = self.ptList[i+1]
            pen = QtGui.QPen(QtGui.QColor(255,0,0), 4, QtCore.Qt.SolidLine)
            painter.setPen(pen)
            painter.drawLine(pt1[0],pt1[1],pt2[0],pt2[1])

    def clearArea(self):
        del self.ptList[:]
        self.repaint()

    def reducePt(self):
        print len(self.ptList)
        self.ptList = self.reducer.rdp(self.ptList, 8)
        print len(self.ptList)
        # self.clearArea()
        self.repaint()


def main():
    app = QtGui.QApplication(sys.argv)
    ex = MainWidget()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()