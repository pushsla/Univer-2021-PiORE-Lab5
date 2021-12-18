import numpy as np

from PyQt5.QtWidgets import QMainWindow, QSpinBox, QPushButton, QVBoxLayout
from PyQt5.uic import loadUi

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

from trees.SimpleDecisonTree import SimpleDecisionTree as SDT


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        loadUi("frontend/main.ui", self)

        self.spinBoxSamples: QSpinBox = self.spinBoxSamples
        self.spinBoxClasses: QSpinBox = self.spinBoxClasses
        self.spinBoxFeatures: QSpinBox = self.spinBoxFeatures
        self.spinBoxTreeSamples: QSpinBox = self.spinBoxTreeSamples
        self.pushButtonTrain: QPushButton = self.pushButtonTrain
        self.pushButtonTest: QPushButton = self.pushButtonTest
        self.pushButtonRegen: QPushButton = self.pushButtonRegen
        self.verticalLayoutPlot: QVBoxLayout = self.verticalLayoutPlot

        self.samples = 3
        self.classes = 1
        self.features = 2
        self.treesamples = 2
        self.tree = SDT()

        self.min_feature_value = 0
        self.max_feature_value = 10

        self.class_markers = {
            0: "*",
            1: "^",
            2: "o",
            3: '+',
            None: "1"
        }

        self.class_colors = {
            0: "red",
            1: 'green',
            2: 'blue',
            3: 'cyan',
            None: 'grey'
        }

        self.test_class_colors = {
            0: "#ffbbbb",
            1: '#bbffbb',
            2: '#bbbbff',
            3: '#bbffff',
            None: 'grey'
        }

        self.x = np.array([])
        self.y = np.array([])

        self.__data_fig: plt.figure = None

        self._connect_matplotlib()
        self._connect_data_slots()
        self._connect_event_clots()

    def reset_plot(self):
        self.__data_fig.clear()

        sample_per_class = int(self.samples/self.classes)

        if self.features == 2:
            self.ax = self.__data_fig.add_subplot()
        elif self.features == 3:
            self.ax = self.__data_fig.add_subplot(projection="3d")

        for c in range(self.classes):
            xc = self.x[np.where(self.y == c)[0]]
            self.ax.scatter(*xc.T, marker=self.class_markers[c], c=self.class_colors[c])

        self.__data_canvas.draw()


    def reset_tree(self):
        self.tree = SDT(feature_split_intervals=self.treesamples)

    def reset_dataset(self):

        classes_centers = np.array([
            [np.random.randint(self.min_feature_value, self.max_feature_value) for _ in range(self.features)]
            for __ in range(self.classes)
        ])
        samples_per_class = self.samples//self.classes
        class_deviation = ((self.max_feature_value-self.min_feature_value)//self.classes)*0.4

        x = []
        y = []
        for c in range(self.classes):
            for _ in range(samples_per_class):
                y.append([c])
                center = classes_centers[c]
                x.append([np.random.uniform(center[i]-class_deviation, center[i]+class_deviation) for i in range(self.features)])

        self.y = np.array(y)
        self.x = np.array(x)

        self.reset_plot()

    def samples_changed(self):
        self.samples = self.spinBoxSamples.value()
        self.reset_dataset()

    def classes_changed(self):
        self.classes = self.spinBoxClasses.value()
        self.reset_dataset()

    def features_changed(self):
        self.features = self.spinBoxFeatures.value()
        self.reset_dataset()

    def treesamples_changed(self):
        self.treesamples = self.spinBoxTreeSamples.value()
        self.reset_tree()

    def train(self):
        self.tree.fit(self.x, self.y)

        self.tree.make_dot()

    def test(self):
        for _ in range(100):
            dot = np.array([np.random.uniform(self.min_feature_value, self.max_feature_value) for _ in range(self.features)])
            prediction = self.tree.predict(dot)

            self.ax.scatter(*dot, marker=self.class_markers[prediction], c=self.test_class_colors[prediction])
        self.__data_canvas.draw()

    def update_all_values(self):
        self.samples_changed()
        self.classes_changed()
        self.features_changed()
        self.treesamples_changed()

    def _connect_data_slots(self):
        self.update_all_values()

    def _connect_event_clots(self):
        self.spinBoxClasses.valueChanged.connect(self.classes_changed)
        self.spinBoxSamples.valueChanged.connect(self.samples_changed)
        self.spinBoxTreeSamples.valueChanged.connect(self.treesamples_changed)
        self.spinBoxFeatures.valueChanged.connect(self.features_changed)

        self.pushButtonRegen.clicked.connect(self.reset_dataset)
        self.pushButtonTrain.clicked.connect(self.train)
        self.pushButtonTest.clicked.connect(self.test)

    def _connect_matplotlib(self):
        self.__data_fig = plt.figure()
        self.ax = self.__data_fig.add_subplot()
        self.__data_canvas = FigureCanvas(self.__data_fig)
        self.verticalLayoutPlot.addWidget(self.__data_canvas)