# -*- coding: utf-8 -*-

import sys
from PyQt4 import QtGui, QtCore
from math import sqrt
import numpy as np
from net import NeuralNetwork

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
        result = []
        for i in range(0, len(points)):
            if(self.weights[i] >= maxTolerace):
                result.append(points[i])
        return result

    def pointNormalize(self, points):
        xmax = max(x for x,y in points)
        xmin = min(x for x,y in points)
        ymax = max(y for x,y in points)
        ymin = min(y for x,y in points)

        Xm = (xmax + xmin) * 0.5 
        Ym = (ymax + ymin) * 0.5
        Sx = (xmax - xmin) * 0.5
        Sy = (ymax - ymin) * 0.5
        Smax = max(Sx, Sy)
        result = []
        for i in range(0, len(points)):
            x = int(50 + 50*((points[i][0] - Xm) / Sx))
            # 100 т.к в обучающей выборке цифры перевернуты
            y = 100-int(50 + 50*((points[i][1] - Ym) / Sy))
            result.append([x, y])
        return result

    def pointsReshape(self, points):
        tmp = np.asarray(points)
        result = np.reshape(tmp, 16)
        return result

class MainWidget(QtGui.QWidget):
    reducer = 0
    ptList = []
    mouseLocation = [0, 0]
    lastPos = [0, 0]
    isPainting = False
    isPainted = False
    clearButton = 0
    recogButton = 0
    nnet = None
    digit = None

    def __init__(self):
        super(MainWidget, self).__init__()
        self.reducer = Reducer()
        self.nnet = NeuralNetwork()
        self.nnet.load()
        self.ptList = []
        self.mouseLocation = [0, 0]
        self.lastPos = [0, 0]
        self.isPainting = False
        self.isPainted = False
        self.clearButton = 0
        self.recogButton = 0
        self.digit = []
        self.initUI()

    def initUI(self):
        self.clearButton = QtGui.QPushButton("Clear")
        QtCore.QObject.connect(self.clearButton, QtCore.SIGNAL("clicked()"),self.clearArea)
        self.recogButton = QtGui.QPushButton("Recognize")
        QtCore.QObject.connect(self.recogButton, QtCore.SIGNAL("clicked()"),self.recog)

        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(self.clearButton)
        hbox.addWidget(self.recogButton)
        vbox = QtGui.QVBoxLayout()
        vbox.addStretch(1)
        vbox.addLayout(hbox)
        self.setLayout(vbox)

        self.setGeometry(300, 300, 500, 500)
        self.show()

    def recog(self):
        arr = self.reducer.pointsReshape(self.ptList)
        print self.nnet.predict(arr)

    def mousePressEvent(self, event):
        self.isPainting = True
        self.isPainted = False

    def mouseMoveEvent(self, event):
        if(self.isPainting == True):
            self.mouseLocation = [event.x(), event.y()]
            if(self.lastPos[0] != self.mouseLocation[0] and self.lastPos[1] != self.mouseLocation[1]):
                self.lastPos = [event.x(), event.y()]
                self.ptList.append([event.x(), event.y()])
                self.repaint()

    def mouseReleaseEvent(self, event):
        self.isPainting = False
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
        # ar =   [[99, 80], [63,100], [25, 76], [79, 68],[100, 62], [97, 23], [54,  0],  [0, 16]]
        # self.clearArea()
        # self.ptList = ar
        self.ptList = self.reducer.rdp(self.ptList, 8)
        # self.repaint()
        self.ptList = self.reducer.pointNormalize(self.ptList)
        self.repaint()

def main():
    app = QtGui.QApplication(sys.argv)
    ex = MainWidget()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()