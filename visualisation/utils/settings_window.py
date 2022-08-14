from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QPalette, QPainter
from PyQt5.QtWidgets import QLabel, QSizePolicy, QMessageBox, QMainWindow, QMenu, QAction, QCheckBox, QComboBox, \
    qApp, QWidget, QGroupBox, QFormLayout, QLineEdit, QComboBox, QSpinBox, QVBoxLayout, QHBoxLayout, QPushButton, \
    QDoubleSpinBox
from PyQt5.QtCore import Qt
import numpy as np

class SettingsWindow(QWidget):
    def __init__(self, parent, settings=None):
        super().__init__(parent)
        self.setWindowTitle("Настройки приложения")
        self.setWindowFlag(Qt.Tool)

        self.settings = {}

        print(settings)

        # настройки темы

        self.formGroupBoxGlobal = QGroupBox("Настройки приложения")

        layout_global = QFormLayout()

        self.theme_combo = QComboBox()
        self.themes = np.array(['dark_amber.xml',
                                'dark_blue.xml',
                                'dark_cyan.xml',
                                'dark_lightgreen.xml',
                                'dark_pink.xml',
                                'dark_purple.xml',
                                'dark_red.xml',
                                'dark_teal.xml',
                                'dark_yellow.xml',
                                'light_amber.xml',
                                'light_blue.xml',
                                'light_cyan.xml',
                                'light_cyan_500.xml',
                                'light_lightgreen.xml',
                                'light_pink.xml',
                                'light_purple.xml',
                                'light_red.xml',
                                'light_teal.xml',
                                'light_yellow.xml'])

        self.themes_rus_names = np.array(['темно-янтарный',
                                          'темно-синий',
                                          'темно-голубой',
                                          'темно-светло-зеленый',
                                          'темно-розовый',
                                          'темно фиолетовый',
                                          'темно-красный',
                                          'темно-бирюзовый',
                                          'темно-желтый',
                                          'светлый янтарь',
                                          'светло-синий',
                                          'светлый циан',
                                          'светлый циан 500',
                                          'светло-зеленый',
                                          'светло-розовый',
                                          'светло-фиолетовый',
                                          'светло-красный',
                                          'светло-бирюзовый',
                                          'светло-желтый'])

        self.theme_combo.addItems(self.themes_rus_names)
        theme_label = QLabel("Тема приложения:")
        layout_global.addRow(theme_label, self.theme_combo)

        if settings:
            idx = np.where(self.themes == settings["theme"])[0][0]
            self.theme_combo.setCurrentIndex(idx)

        self.formGroupBoxGlobal.setLayout(layout_global)

        # настройки параметров модели

        self.formGroupBoxSim = QGroupBox("Настройки моделирования")
        layout_sim = QFormLayout()

        self.n_spin = QSpinBox()
        self.n_spin.setMinimum(1)
        if settings:
            self.n_spin.setValue(settings['n'])
        else:
            self.n_spin.setValue(3)

        self.n_label = QLabel("Число каналов:")

        self.r_spin = QSpinBox()
        self.r_spin.setMinimum(0)
        if settings:
            self.r_spin.setValue(settings['r'])
        else:
            self.r_spin.setValue(10)

        self.r_label = QLabel("Размер буфера:")

        self.ro_spin = QDoubleSpinBox()
        self.ro_spin.setDecimals(3)
        if settings:
            self.ro_spin.setValue(settings['ro'])
        else:
            self.ro_spin.setValue(0.7)
        self.ro_spin.setMinimum(0.01)
        self.ro_spin.setMaximum(0.99)
        self.ro_spin.setSingleStep(0.01)

        layout_sim.addRow(self.n_label, self.n_spin)
        layout_sim.addRow(self.r_label, self.r_spin)
        layout_sim.addRow(QLabel("Коэффициент загрузки:"), self.ro_spin)

        self.formGroupBoxSim.setLayout(layout_sim)



        btnLayout = QHBoxLayout()

        self.okBtn = QPushButton('Принять', self)
        self.okBtn.clicked.connect(self.on_ok_clicked)

        self.cancelBtn = QPushButton('Отменить', self)
        self.cancelBtn.clicked.connect(self.on_cancel_clicked)

        btnLayout.addWidget(self.okBtn)
        btnLayout.addWidget(self.cancelBtn)

        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.formGroupBoxGlobal)
        mainLayout.addWidget(self.formGroupBoxSim)
        mainLayout.addLayout(btnLayout)
        self.setLayout(mainLayout)

        self.resize(500, 500)


    def on_ok_clicked(self):
        self.settings['theme'] = self.themes[self.theme_combo.currentIndex()]
        self.settings['n'] = self.n_spin.value()
        self.settings['r'] = self.r_spin.value()
        self.settings['ro'] = self.ro_spin.value()

        self.close()

    def on_cancel_clicked(self):
        self.settings.clear()
        self.close()
