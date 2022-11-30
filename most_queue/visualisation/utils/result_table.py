from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QPalette, QPainter
from PyQt5.QtWidgets import QLabel, QSizePolicy, QMessageBox, QMainWindow, QMenu, QAction, QCheckBox, QComboBox, \
    qApp, QWidget, QGroupBox, QFormLayout, QLineEdit, QComboBox, QSpinBox, QVBoxLayout, QHBoxLayout, QPushButton, \
    QDoubleSpinBox, QTableWidget, QTableWidgetItem
from PyQt5.QtCore import Qt, QSize
import numpy as np


class ResultsTableWindow(QWidget):
    def __init__(self, parent, detect_info=None):
        super().__init__(parent)
        self.setWindowTitle("Результаты обнаружения объектов")
        self.setWindowFlag(Qt.Tool)

        self.detect_info = detect_info

        mainLayout = QVBoxLayout()
        mainLayout.setAlignment(Qt.AlignJustify)

        if len(self.detect_info) != 0:
            table = QTableWidget()  # Create a table
            table.setColumnCount(2)
            table.setRowCount(len(self.detect_info))  # and one row


            table.setHorizontalHeaderLabels(["Наименование", "Вероятность"])

            table.horizontalHeaderItem(0).setToolTip("Наименование")
            table.horizontalHeaderItem(1).setToolTip("Вероятность")

            # Set the alignment to the headers
            table.horizontalHeaderItem(0).setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            table.horizontalHeaderItem(1).setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

            # Fill the first line
            row_num = 0
            for row in self.detect_info:
                for j in range(2):
                    item = QTableWidgetItem(str(row[j]))
                    item.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                    table.setItem(row_num, j, item)
                row_num += 1

            # Do the resize of the columns by content

            table.setMinimumWidth(500)
            table.resizeColumnsToContents()

            mainLayout.addWidget(table)

        # btnLayout = QHBoxLayout()
        #
        # self.okBtn = QPushButton('Принять', self)
        # self.okBtn.clicked.connect(self.on_ok_clicked)
        #
        # self.cancelBtn = QPushButton('Отменить', self)
        # self.cancelBtn.clicked.connect(self.on_cancel_clicked)
        #
        # btnLayout.addWidget(self.okBtn)
        # btnLayout.addWidget(self.cancelBtn)
        # mainLayout.addLayout(btnLayout)
        self.setLayout(mainLayout)

        self.resize(600, 400)

    def on_ok_clicked(self):
        # self.settings['conf_thres'] = self.conf_thres_spin.value()
        # self.settings['iou_thres'] = self.IOU_spin.value()
        # self.settings['CNN'] = self.cnn_combo.currentText()
        # self.settings['theme'] = self.themes[self.theme_combo.currentIndex()]
        # self.settings['Seg model'] = self.seg_combo.currentText()
        # self.settings['k_means_clusters'] = self.k_means_clusters_spin.value()
        self.close()

    def on_cancel_clicked(self):
        # self.settings.clear()
        self.close()
