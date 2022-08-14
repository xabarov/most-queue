import sys, time
from PyQt5.QtCore import *
from PyQt5.QtGui import QPixmap, QMovie, QPainter
from PyQt5.QtWidgets import QSplashScreen, QWidget, QLabel, QVBoxLayout
from PyQt5 import QtCore

class MovieSplashScreen(QSplashScreen):

    def __init__(self, movie, parent=None, parent_geo=None, size_set=None, **args):

        if size_set:
            size = QtCore.QSize(size_set[0], size_set[1])
            movie.setScaledSize(size)

        movie.jumpToFrame(0)
        self.pixmap = QPixmap(movie.frameRect().size())
        QSplashScreen.__init__(self, pixmap=self.pixmap, **args)

        if parent_geo:
            self.desktop_geo = parent_geo
            self.fit()

        self.movie = movie
        self.movie.frameChanged.connect(self.repaint)

    def fit(self):
        x = self.desktop_geo.topLeft().x()
        y = self.desktop_geo.topLeft().y()
        w = self.desktop_geo.width()
        h = self.desktop_geo.height()

        self.move(x+w//2 - self.width()//2, y+h//2 - self.height()//2)

    def showEvent(self, event):
        self.movie.start()

    def hideEvent(self, event):
        self.movie.stop()

    def paintEvent(self, event):
        painter = QPainter(self)
        pixmap = self.movie.currentPixmap()
        self.setMask(pixmap.mask())
        painter.drawPixmap(0, 0, pixmap)

    def sizeHint(self):
        return self.movie.scaledSize()


