#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PyQt5.QtGui import QPainter, QMovie
from PyQt5.QtWidgets import QMessageBox, QVBoxLayout, QMainWindow, QMenu, \
    QAction, QToolBar, QWidget, QGridLayout
from PyQt5.QtCore import Qt
from PyQt5 import QtCore
from PyQt5.QtGui import QIcon, QColor, QPen, QBrush
from settings_window import SettingsWindow
# Display image
from PyQt5.QtWidgets import qApp
from splash_screen import MovieSplashScreen

qApp.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
from most_queue.sim.smo_im import SmoIm
from most_queue.sim.smo_im_prty import SmoImPrty
import time
import math

COLORS = {"dark_amber.xml": (255, 215, 64, 255),
          "dark_blue.xml": (68, 138, 255, 255),
          "dark_cyan.xml": (77, 208, 225, 255),
          "dark_lightgreen.xml": (139, 195, 74, 255),
          "dark_pink.xml": (255, 64, 129, 255),
          "dark_purple.xml": (171, 71, 188, 255),
          "dark_red.xml": (255, 23, 68, 255),
          "dark_teal.xml": (29, 233, 182, 255),
          "dark_yellow.xml": (255, 255, 0, 255),
          "light_amber.xml": (255, 196, 0, 255),
          "light_blue.xml": (41, 121, 255, 255),
          "light_blue_500.xml": (3, 169, 244, 255),
          "light_cyan.xml": (0, 229, 255, 255),
          "light_cyan_500.xml": (0, 188, 212, 255),
          "light_lightgreen.xml": (100, 221, 23, 255),
          "light_lightgreen_500.xml": (139, 195, 74, 255),
          "light_orange.xml": (255, 61, 0, 255),
          "light_pink.xml": (255, 64, 129, 255),
          "light_pink_500.xml": (233, 30, 99, 255),
          "light_purple.xml": (224, 64, 251, 255),
          "light_purple_500.xml": (156, 39, 176, 255),
          "light_red.xml": (255, 23, 68, 255),
          "light_red_500.xml": (244, 67, 54, 255),
          "light_teal.xml": (29, 233, 182, 255),
          "light_teal_500.xml": (0, 150, 136, 255),
          "light_yellow.xml": (255, 234, 0, 255)}

COLOR_LIST = []
for key in COLORS:
    COLOR_LIST.append(COLORS[key])

class SmoThread(QtCore.QThread):
    mysignal = QtCore.pyqtSignal(str)

    def __init__(self, parent, n, source_params, server_params, jobs_count, r=None, k=3, model="FCFS один класс",
                 speed=50):
        QtCore.QThread.__init__(self, parent)

        self.n = n
        self.k = k
        self.model_type = model
        self.source_params = source_params
        self.server_params = server_params
        self.jobs_count = jobs_count
        self.iters_to_end = jobs_count
        self.r = r
        self.iters_left = 0
        self.is_running = False
        self.is_smo_created = False
        self.speed = speed

    def calc_speed_to_sleep(self, speed, time_to_next_event, factor=0.3):

        max_value = 3  # sec
        min_value = 0.1  # sec
        a = -math.log(min_value / max_value) / 100
        delay = max_value * math.exp(-a * factor*speed/time_to_next_event)
        return delay

    def run(self):
        self.is_running = True

        if not self.is_smo_created:
            if self.model_type == "FCFS один класс":
                self.smo = SmoIm(self.n, buffer=self.r, calc_next_event_time=True)

                self.smo.set_sources(self.source_params["params"], self.source_params["type"])
                self.smo.set_servers(self.server_params["params"], self.server_params["type"])
            else:
                self.smo = SmoImPrty(self.n, self.k, self.model_type, buffer=self.r, calc_next_event_time=True)
                sources = []
                servers_params = []
                for j in range(self.k):
                    sources.append({'type': self.source_params["type"], 'params': self.source_params["params"]})
                    servers_params.append({'type': self.server_params["type"], 'params': self.server_params["params"]})

                self.smo.set_sources(sources)
                self.smo.set_servers(servers_params)

            self.is_smo_created = True

        delta = self.iters_left
        for i in range(self.iters_to_end):
            if self.is_running:
                start_iter = i + 1 + delta
                end_iter = self.jobs_count
                print("Step {0:d} from {1:d}, remaining {2:d}".format(start_iter, end_iter, self.iters_to_end - 1))
                self.smo.run_one_step()
                self.iters_left += 1
                self.iters_to_end -= 1

                time_to_next_event = self.smo.time_to_next_event
                # w_im = self.smo.w

                # print("\nЗначения начальных моментов времени ожидания заявок в системе:\n")
                #
                # print("{0:^15s}|{1:^15s}".format("№ момента", "ИМ"))
                # print("-" * 45)
                # for j in range(3):
                #     print("{0:^16d}|{1:^15.5g}".format(j + 1, w_im[j]))
                # print("\n\nДанные ИМ::\n")
                # print(smo)
                if self.model_type == "FCFS один класс":
                    params = str(len(self.smo.queue)) + ","
                else:
                    params = ""
                    for j in range(self.k):
                        params += str(len(self.smo.queue[j])) + ","

                for s in self.smo.servers:
                    if s.is_free:
                        params += "0,"
                    else:
                        if self.model_type == "FCFS один класс":
                            params += "1,"
                        else:
                            params += str(s.class_on_service + 1) + ","

                self.mysignal.emit(params)
                time.sleep(self.calc_speed_to_sleep(self.speed, time_to_next_event))


class QueueWidget(QWidget):

    def __init__(self, queue_count=10, number_of_jobs=0, theme="dark_blue.xml", k=1, model="FCFS один класс"):
        super().__init__()

        self.queue_count = queue_count
        self.number_of_jobs = number_of_jobs
        self.theme = theme

        self.k = k
        self.model = model

        self.initUI()

    def initUI(self):
        self.show()

    def paintEvent(self, e):
        qp = QPainter()
        qp.begin(self)
        self.drawLines(qp)
        qp.end()

    def drawLines(self, qp):

        # self.queue_count = min(10, self.queue_count)
        self.color = COLORS[self.theme]

        if "dark" in self.theme:
            pen_color = QColor("#F6F7F2")
        else:
            pen_color = Qt.black

        pen = QPen(pen_color, 2, Qt.DashLine)

        width = self.width()
        height = self.height()

        step = int(width / self.queue_count)

        x = 0

        paddings = int((height - step) / 2)
        bottom = paddings + step

        if self.k == 1:
            for i in range(self.queue_count + 1):
                qp.setPen(pen)
                qp.drawLine(x, paddings, x, bottom)
                qp.drawLine(0, paddings, self.queue_count * step, paddings)
                qp.drawLine(0, bottom, self.queue_count * step, bottom)

                if i >= self.queue_count - self.number_of_jobs and i != self.queue_count:
                    qp.setRenderHint(QPainter.Antialiasing)
                    qp.setPen(QPen(QColor(*self.color), 1, Qt.SolidLine))
                    qp.setBrush(QBrush(QColor(*self.color), Qt.SolidPattern))
                    qp.drawEllipse(x, paddings, step,
                                   step)

                x += step
        else:
            jobs_paint = 0
            for j in range(self.k):
                for i in range(self.number_of_jobs[j]):
                    if jobs_paint <= self.queue_count:
                        qp.setPen(pen)
                        qp.drawLine(self.queue_count * step - x, paddings, self.queue_count * step - x, bottom)
                        qp.drawLine(self.queue_count * step - x, paddings, (self.queue_count - 1) * step - x, paddings)
                        qp.drawLine(self.queue_count * step - x, bottom, (self.queue_count - 1) * step - x, bottom)

                        qp.setRenderHint(QPainter.Antialiasing)
                        qp.setPen(QPen(QColor(*COLOR_LIST[j+1]), 1, Qt.SolidLine))
                        qp.setBrush(QBrush(QColor(*COLOR_LIST[j+1]), Qt.SolidPattern))
                        qp.drawEllipse(self.queue_count * step - x, paddings, step,
                                       step)
                        jobs_paint += 1
                        x += step

            for job in range(self.queue_count - jobs_paint + 1):
                qp.setPen(pen)
                qp.drawLine(self.queue_count * step - x, paddings, self.queue_count * step - x, bottom)
                qp.drawLine(self.queue_count * step - x, paddings, (self.queue_count - 1) * step - x, paddings)
                qp.drawLine(self.queue_count * step - x, bottom, (self.queue_count - 1) * step - x, bottom)
                x += step


class ChannelsWidget(QWidget):

    def __init__(self, channels_count=5, channels_in_service=None, theme="dark_blue.xml", k=1, model="FCFS один класс"):
        super().__init__()

        self.channels_count = channels_count
        self.channels_in_service = channels_in_service
        self.theme = theme

        self.k = k
        self.model = model
        self.initUI()

    def initUI(self):
        self.show()

    def paintEvent(self, e):
        qp = QPainter()
        qp.begin(self)
        self.drawLines(qp)
        qp.end()

    def drawLines(self, qp):
        self.color = COLORS[self.theme]

        if "dark" in self.theme:
            pen_color = QColor("#F6F7F2")
        else:
            pen_color = Qt.black

        pen = QPen(pen_color, 2, Qt.DashLine)

        width = self.width()
        height = self.height()

        fract = 1.0 / 3

        if self.channels_count != 1:
            channel_height = int(height / (self.channels_count + (self.channels_count - 1) * fract))
            x = 0
        else:
            channel_height = int(height * fract)
            x = channel_height

        channel_width = int(2 * fract * width)

        paddings_left = int((width - channel_width) / 2)
        paddings_right = paddings_left + channel_width

        qp.setPen(pen)

        for i in range(self.channels_count):
            qp.setPen(pen)
            qp.drawLine(paddings_left, x, paddings_right, x)
            qp.drawLine(paddings_left, x + channel_height, paddings_right, x + channel_height)
            qp.drawLine(paddings_left, x, paddings_left, x + channel_height)
            qp.drawLine(paddings_right, x, paddings_right, x + channel_height)

            if self.channels_in_service:
                if self.channels_in_service[i] != 0:

                    if self.k == 1:
                        color_set = QColor(*self.color)
                    else:
                        color_set = QColor(*COLOR_LIST[self.channels_in_service[i]])

                    qp.setRenderHint(QPainter.Antialiasing)
                    qp.setPen(QPen(color_set, 1, Qt.SolidLine))
                    qp.setBrush(QBrush(color_set, Qt.SolidPattern))
                    if channel_width > channel_height:
                        delta = int((channel_width - channel_height) / 2)
                        qp.drawEllipse(paddings_left + delta, x, min(channel_width, channel_height),
                                       min(channel_width, channel_height))
                    else:
                        delta = int((channel_height - channel_width) / 2)
                        qp.drawEllipse(paddings_left, x + delta, min(channel_width, channel_height),
                                       min(channel_width, channel_height))

            x += channel_height + int(0.3 * channel_height)


class SmoVisualizationWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        n = 5
        r = 10
        l = 1.0
        ro = 0.9
        jobs_count = 10000
        source_params = {}
        server_params = {}
        source_params["type"] = "M"
        server_params["type"] = "M"

        source_params["params"] = l
        mu = l / (n * ro)

        server_params["params"] = mu

        self.n = n
        self.k = 1
        self.model_type = 'FCFS один класс'
        self.source_params = source_params
        self.server_params = server_params
        self.jobs_count = jobs_count
        self.r = r
        self.in_queue = 0
        self.ro = self.calc_ro()

        # Установка темы оформления
        self.theme_str = 'dark_blue.xml'
        self.is_dark_theme = True

        # заставка
        self.start_gif(is_prog_load=True)

        # Иконка
        self.setWindowIcon(QIcon("icons/planet-earth.png"))

        self.create_widgets()

        # создаем меню и тулбар
        self.createActions()
        self.createMenus()
        self.createToolbar()
        self.app_settings_set = False

        # заголовок главного окна
        self.setWindowTitle("Визуализация работы СМО")

        # инициализация настроек приложения
        self.init_app_settings()

        self.splash.finish(self)

        self.is_running = False
        self.is_resume_mode = False

    def change_server_params(self, server_type, server_coev):

        if server_type == "M":
            self.server_params["params"] = 1.0 / self.b1

        elif server_type == "D":
            self.server_params["params"] = self.b1
        elif server_type == "Uniform":
            self.server_params["params"] = [self.b1]

        elif server_type == "H":

            self.server_params["params"] = rd.H2_dist.get_params_by_mean_and_coev(self.b1, server_coev)

        elif server_type == "Gamma":
            self.server_params["params"] = rd.Gamma.get_mu_alpha_by_mean_and_coev(self.b1, server_coev)

        elif server_type == "E":
            self.server_params["params"] = rd.Erlang_dist.get_params_by_mean_and_coev(self.b1, server_coev)

        elif server_type == "Pa":
            self.server_params["params"] = rd.Pareto_dist.get_a_k_by_mean_and_coev(self.b1, server_coev)

        self.server_params["type"] = server_type

        print("Новые параметры распр. времени обсл. потока:")
        print("    ", self.server_params)

    def change_source_params(self, source_type, source_coev, k_new=1, model='FCFS один класс'):

        coev_mul = 1

        if self.k != k_new:
            coev_mul = self.k / k_new

        if source_type == "M":
            self.source_params["params"] = self.l * coev_mul

        elif source_type == "D":
            self.source_params["params"] = 1.0 / (self.l * coev_mul)

        elif source_type == "Uniform":
            self.source_params["params"] = [1.0 / (self.l * coev_mul)]

        elif source_type == "H":
            self.source_params["params"] = rd.H2_dist.get_params_by_mean_and_coev(1.0 / (self.l * coev_mul),
                                                                                  source_coev)

        elif source_type == "Gamma":
            self.source_params["params"] = rd.Gamma.get_mu_alpha_by_mean_and_coev(1.0 / (self.l * coev_mul),
                                                                                  source_coev)

        elif source_type == "E":
            self.source_params["params"] = rd.Erlang_dist.get_params_by_mean_and_coev(1.0 / (self.l * coev_mul),
                                                                                      source_coev)

        elif source_type == "Pa":
            self.source_params["params"] = rd.Pareto_dist.get_a_k_by_mean_and_coev(1.0 / (self.l * coev_mul),
                                                                                   source_coev)

        self.source_params["type"] = source_type

        print("Новые параметры вх потока:")
        print("    ", self.source_params)

    def change_ro(self, ro_new):
        if self.ro == ro_new:
            return

        l = 1.0

        if self.source_params["type"] == "M":
            l = self.source_params["params"]
        elif self.source_params["type"] == "D":
            l = 1.00 / self.source_params["params"]
        elif self.source_params["type"] == "Uniform":
            l = 1.00 / self.source_params["params"][0]
        elif self.source_params["type"] == "H":
            y1 = self.source_params[0]
            y2 = 1.0 - self.source_params["params"][0]
            mu1 = self.source_params["params"][1]
            mu2 = self.source_params["params"][2]

            f1 = y1 / mu1 + y2 / mu2
            l = 1.0 / f1

        elif self.source_params["type"] == "E":
            r = self.source_params["params"][0]
            mu = self.source_params["params"][1]
            l = mu / r

        elif self.source_params["type"] == "Gamma":
            mu = self.source_params["params"][0]
            alpha = self.source_params["params"][1]
            l = mu / alpha

        elif self.source_params["type"] == "Pa":
            if self.source_params["params"][0] < 1:
                return None
            else:
                a = self.source_params["params"][0]
                k = self.source_params["params"][1]
                f1 = a * k / (a - 1)
                l = 1.0 / f1

        if self.server_params["type"] == "M":
            self.b1 = ro_new * self.n / l
            self.server_params["params"] = 1.0 / self.b1

        elif self.server_params["type"] == "D" or self.server_params["type"] == "Uniform":
            self.b1 = ro_new * self.n / l
            self.server_params["params"] = self.b1

        elif self.server_params["type"] == "H":
            y1 = self.server_params["params"][0]
            mu1 = self.server_params["params"][1]
            mu2 = self.server_params["params"][2]

            b = rd.H2_dist.calc_theory_moments(y1, mu1, mu2)
            b[0] = ro_new * self.n / l

            self.b1 = b[0]
            coev = math.sqrt(b[1] - b[0] ** 2) / b[0]
            params_new = rd.H2_dist.get_params_by_mean_and_coev(b[0], coev)
            self.server_params["params"] = params_new

        elif self.server_params["type"] == "Gamma":
            mu = self.server_params["params"][0]
            alpha = self.server_params["params"][1]
            b = rd.Gamma.calc_theory_moments(mu, alpha)
            b[0] = ro_new * self.n / l
            self.b1 = b[0]

            coev = math.sqrt(b[1] - b[0] ** 2) / b[0]
            params_new = rd.Gamma.get_mu_alpha_by_mean_and_coev(b[0], coev)
            self.server_params["params"] = params_new

        elif self.server_params["type"] == "E":
            r = self.server_params["params"][0]
            mu = self.server_params["params"][1]
            b = rd.Erlang_dist.calc_theory_moments(r, mu)
            b[0] = ro_new * self.n / l
            self.b1 = b[0]

            coev = math.sqrt(b[1] - b[0] ** 2) / b[0]
            params_new = rd.Erlang_dist.get_params_by_mean_and_coev(b[0], coev)
            self.server_params["params"] = params_new

        elif self.server_params["type"] == "Pa":
            if self.server_params["params"][0] < 1:
                return math.inf
            else:
                a = self.server_params["params"][0]
                k = self.server_params["params"][1]
                b = rd.Pareto_dist.calc_theory_moments(a, k)
                b[0] = ro_new * self.n / l

                self.b1 = b[0]

                coev = math.sqrt(b[1] - b[0] ** 2) / b[0]
                params_new = rd.Pareto_dist.get_a_k_by_mean_and_coev(b[0], coev)
                self.server_params["params"] = params_new

        print("Коэфф загрузки изменен с {0:f} на {1:f}".format(self.ro, self.calc_ro()))

    def calc_ro(self):
        """
        вычисляет коэффициент загрузки СМО
        """

        l = 0
        if self.source_params["type"] == "M":
            l = self.source_params["params"]
        elif self.source_params["type"] == "D":
            l = 1.00 / self.source_params["params"]
        elif self.source_params["type"] == "Uniform":
            l = 1.00 / self.source_params["params"][0]
        elif self.source_params["type"] == "H":
            y1 = self.source_params[0]
            y2 = 1.0 - self.source_params["params"][0]
            mu1 = self.source_params["params"][1]
            mu2 = self.source_params["params"][2]

            f1 = y1 / mu1 + y2 / mu2
            l = 1.0 / f1

        elif self.source_params["type"] == "E":
            r = self.source_params["params"][0]
            mu = self.source_params["params"][1]
            l = mu / r

        elif self.source_params["type"] == "Gamma":
            mu = self.source_params["params"][0]
            alpha = self.source_params["params"][1]
            l = mu / alpha

        elif self.source_params["type"] == "Pa":
            if self.source_params["params"][0] < 1:
                return None
            else:
                a = self.source_params["params"][0]
                k = self.source_params["params"][1]
                f1 = a * k / (a - 1)
                l = 1.0 / f1

        b1 = 0
        if self.server_params["type"] == "M":
            mu = self.server_params["params"]
            b1 = 1.0 / mu
        elif self.server_params["type"] == "D":
            b1 = self.source_params["params"]
        elif self.server_params["type"] == "Uniform":
            b1 = self.source_params["params"][0]

        elif self.server_params["type"] == "H":
            y1 = self.server_params["params"][0]
            y2 = 1.0 - self.server_params["params"][0]
            mu1 = self.server_params["params"][1]
            mu2 = self.server_params["params"][2]

            b1 = y1 / mu1 + y2 / mu2

        elif self.server_params["type"] == "Gamma":
            mu = self.server_params["params"][0]
            alpha = self.server_params["params"][1]
            b1 = alpha / mu

        elif self.server_params["type"] == "E":
            r = self.server_params["params"][0]
            mu = self.server_params["params"][1]
            b1 = r / mu

        elif self.server_params["type"] == "Pa":
            if self.server_params["params"][0] < 1:
                return math.inf
            else:
                a = self.server_params["params"][0]
                k = self.server_params["params"][1]
                b1 = a * k / (a - 1)

        self.l = l
        self.b1 = b1
        ro = l * b1 / self.n
        print(ro)
        return ro

    def run(self):

        self.pauseAct.setEnabled(True)
        self.stopAct.setEnabled(True)

        if not self.is_resume_mode:
            speed = self.app_settings["speed"]
            self.sim_worker = SmoThread(self, self.n, self.source_params, self.server_params, self.jobs_count, self.r,
                                        self.k, self.model_type,
                                        speed)
            self.sim_worker.mysignal.connect(self.on_change)

            self.sim_worker.started.connect(self.on_sim_started)
            self.sim_worker.finished.connect(self.on_sim_finished)

        self.is_running = True

        if not self.sim_worker.isRunning():
            self.sim_worker.start()

    def pause(self):
        self.is_resume_mode = True

        print("Try to pause simulattion...")
        if self.is_running and self.sim_worker:
            self.sim_worker.is_running = False

    def on_change(self, s):
        params = s.split(",")
        if self.model_type == "FCFS один класс":
            queue = int(params[0])
        else:
            queue = []
            for j in range(self.k):
                queue.append(int(params[j]))
        channels_is_free = [0] * self.n
        for i in range(self.n):
            channels_is_free[i] = int(params[i + self.k])

        self.queue.number_of_jobs = queue
        self.queue.k = self.k
        self.queue.model = self.model_type

        self.queue.repaint()

        self.channels.channels_in_service = channels_is_free
        self.channels.k = self.k
        self.channels.model = self.model_type

        self.channels.repaint()

    def on_sim_started(self):
        print("Simulation started")

    def on_sim_finished(self):
        print("Simulation finished")

    def stop(self):

        self.pauseAct.setEnabled(False)
        self.stopAct.setEnabled(False)

        if self.is_running and self.sim_worker:
            self.sim_worker.is_running = False

        self.is_resume_mode = False
        channels_is_free = [0] * self.n
        for i in range(self.n):
            channels_is_free[i] = 0

        self.queue.number_of_jobs = 0
        self.queue.k = 1
        self.queue.repaint()

        self.channels.channels_in_service = channels_is_free
        self.channels.k = 1
        self.channels.repaint()

    def create_widgets(self):

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        self.queue_widget = QWidget()
        self.channels_widget = QWidget()

        self.mainlayout = QGridLayout(central_widget)

        for w, (r, c) in zip(
                (self.queue_widget, self.channels_widget),
                ((0, 0), (0, 1)),
        ):
            self.mainlayout.addWidget(w, r, c)

        self.mainlayout.setColumnStretch(0, 2)
        self.mainlayout.setColumnStretch(1, 1)

        self.queue_lay = QVBoxLayout(self.queue_widget)
        self.queue = QueueWidget(self.r, self.in_queue)
        self.queue_lay.addWidget(self.queue)

        self.channels_lay = QVBoxLayout(self.channels_widget)
        self.channels_in_service = None
        self.channels = ChannelsWidget(self.n, self.channels_in_service)
        self.channels_lay.addWidget(self.channels)

        self.resize(1200, 800)

    def set_movie_gif(self):
        """
        Установка гифки на заставку
        """
        theme_type = self.theme_str.split('.')[0].split('_')[1]
        if theme_type == "cyan" or theme_type == "blue" or theme_type == "cyan_500":
            self.movie_gif = "icons/loading-63.gif"
        elif theme_type == "red" or theme_type == "yellow" or theme_type == "amber":
            self.movie_gif = "icons/loading-64.gif"
        elif theme_type == "teal" or theme_type == "lightgreen":
            self.movie_gif = "icons/loading-65.gif"
        else:
            # purple
            self.movie_gif = "icons/loading-66.gif"

    def start_gif(self, is_prog_load=False):
        self.set_movie_gif()
        self.movie = QMovie(self.movie_gif)
        if is_prog_load:
            self.splash = MovieSplashScreen(self.movie)
        else:
            self.splash = MovieSplashScreen(self.movie, parent_geo=self.geometry())

        self.splash.setWindowFlags(
            QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.FramelessWindowHint
        )

        self.splash.showMessage(
            "<h1><font color='red'></font></h1>",
            QtCore.Qt.AlignTop | QtCore.Qt.AlignCenter,
            QtCore.Qt.white,
        )

        self.splash.show()

    def about(self):
        """
        Окно о приложении
        """
        QMessageBox.about(self, "О программе визуализации работы СМО",
                          "<p><b>О программе визуализации работы СМО</b></p>"
                          "<p>Тест описания программы</p>")

    def set_icons(self):
        """
        Задать иконки
        """
        # if self.is_dark_theme:
        #     self.icon_folder = "icons/white_icons"
        # else:
        #     self.icon_folder = "icons/dark_icons"

        theme_type = self.theme_str.split('.')[0]

        self.icon_folder = "icons/" + theme_type

        self.exitAct.setIcon(QIcon(self.icon_folder + "/logout.png"))
        self.aboutAct.setIcon(QIcon(self.icon_folder + "/info.png"))
        self.runAct.setIcon(QIcon(self.icon_folder + "/play-button.png"))
        self.pauseAct.setIcon(QIcon(self.icon_folder + "/pause.png"))
        self.stopAct.setIcon(QIcon(self.icon_folder + "/stop-button.png"))
        self.settingsappAct.setIcon(QIcon(self.icon_folder + "/settings.png"))

    def createActions(self):

        """
        Задать действия
        """

        self.exitAct = QAction("Выход", self, shortcut="Ctrl+Q", triggered=self.close)
        self.aboutAct = QAction("О модуле", self, triggered=self.about)
        self.settingsappAct = QAction("Настройки приложения", self, enabled=True, triggered=self.showappSettings)

        self.runAct = QAction("Начать моделирование...", self, shortcut="Ctrl+R", triggered=self.run)
        self.stopAct = QAction("Завершить моделирование...", self, shortcut="Ctrl+B", enabled=False,
                               triggered=self.stop)
        self.pauseAct = QAction("Приостановить моделирование...", self, shortcut="Ctrl+P", enabled=False,
                                triggered=self.pause)

        self.set_icons()

    def createMenus(self):

        """
        Создание меню
        """

        self.fileMenu = QMenu("&Файл", self)
        self.fileMenu.addAction(self.exitAct)

        self.simMenu = QMenu("&Моделирование", self)
        self.simMenu.addAction(self.runAct)
        self.simMenu.addAction(self.pauseAct)
        self.simMenu.addAction(self.stopAct)

        self.settingsMenu = QMenu("Настройки", self)
        self.settingsMenu.addAction(self.settingsappAct)

        self.helpMenu = QMenu("&Помощь", self)
        self.helpMenu.addAction(self.aboutAct)

        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.simMenu)
        self.menuBar().addMenu(self.settingsMenu)
        self.menuBar().addMenu(self.helpMenu)

    def createToolbar(self):

        """
        Создание тулбаров
        """

        # Слева

        toolBar = QToolBar("Панель инструментов", self)
        toolBar.addAction(self.runAct)
        toolBar.addAction(self.pauseAct)
        toolBar.addAction(self.stopAct)
        toolBar.addSeparator()
        toolBar.addAction(self.settingsappAct)
        toolBar.addSeparator()
        toolBar.addAction(self.exitAct)

        self.toolBarLeft = toolBar
        self.addToolBar(Qt.LeftToolBarArea, self.toolBarLeft)

    def showappSettings(self):
        """
        Показать осно с настройками приложения
        """
        self.stop()

        if self.app_settings_set:
            self._settings_window = SettingsWindow(self, self.app_settings)
        else:
            self._settings_window = SettingsWindow(self)
        self._settings_window.okBtn.clicked.connect(self.on_settings_closed)
        self._settings_window.cancelBtn.clicked.connect(self.on_settings_closed)

        self._settings_window.show()

    def on_settings_closed(self):
        """
        При закрытии окна настроек приложения
        Осуществляет сохранение настроек
        """

        if len(self._settings_window.settings) != 0:
            self.app_settings = self._settings_window.settings
            self.app_settings_set = True
            print("Настройки сохранены:")
            str_settings = "Настройки сохранены:\n"

            for key in self.app_settings:
                print(key, self.app_settings[key])
                str_settings += "{0} {1} ".format(key, self.app_settings[key])

            if self.app_settings['theme'] != self.theme_str:
                print("Start change theme to " + self.app_settings['theme'])
                self.change_theme(self.app_settings['theme'])
                self.theme_str = self.app_settings['theme']
                self.set_icons()

            self.n = self.app_settings['n']
            self.r = self.app_settings['r']

            self.change_ro(self.app_settings['ro'])

            self.change_source_params(self.app_settings["source"], self.app_settings["source_coev"],
                                      self.app_settings['k'], self.app_settings['model'])
            self.change_server_params(self.app_settings["server"], self.app_settings["server_coev"])

            self.k = self.app_settings['k']
            self.model_type = self.app_settings['model']

            self.channels.channels_count = self.n
            channels_is_free = [0] * self.n
            for i in range(self.n):
                channels_is_free[i] = 0

            self.channels.theme = self.app_settings['theme']
            self.channels.channels_in_service = channels_is_free
            self.channels.k = 1
            self.channels.repaint()
            self.queue.queue_count = self.r
            self.queue.theme = self.app_settings['theme']
            self.queue.k = 1
            self.queue.repaint()

            QMessageBox.about(self, "Сохранение настроек приложения",
                              "Установлены новые настройки приложения")

    def change_theme(self, theme_str):
        """
        Изменение темы приложения
        """
        app = QApplication.instance()
        apply_stylesheet(app, theme=theme_str)

    def init_app_settings(self):
        """
        Инициализация настроек приложения
        """
        self.app_settings = {}
        self.app_settings["theme"] = self.theme_str
        self.app_settings["n"] = self.n
        self.app_settings["k"] = self.k
        self.app_settings["r"] = self.r
        self.app_settings["ro"] = self.ro
        self.app_settings["source"] = "M"
        self.app_settings["source_coev"] = 1.0
        self.app_settings["server"] = "M"
        self.app_settings["server_coev"] = 1.0
        self.app_settings["speed"] = 50
        self.app_settings["model"] = 'FCFS один класс'
        self.app_settings_set = True


if __name__ == '__main__':
    import sys
    import most_queue.sim.rand_destribution as rd
    from PyQt5.QtWidgets import QApplication
    from qt_material import apply_stylesheet

    app = QApplication(sys.argv)
    apply_stylesheet(app, theme='dark_blue.xml')

    imageViewer = SmoVisualizationWindow()
    imageViewer.show()
    sys.exit(app.exec_())
