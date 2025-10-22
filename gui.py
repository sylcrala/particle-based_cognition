
from PyQt6 import QtWidgets, QtCore, QtGui

from apis.api_registry import api
from shared_services import logging

# TODO:
## 1. build out GUI frame/layout
#### - rebuild main window
#### - rebuild visualizer
#### - rebuild chat window
###### - add option to export chat logs
###### - add option to view previous chat sessions (no "loading" - all sessions are contextually persistent in memory)
#### - clean up design through rebuild

## 2. build diagnostics APIs (modularized, able to be called independently from anywhere if needed) 
#### - log viewer (with detailed filtering options)
#### - system health monitor (CPU, memory, disk, network, etc)
#### - various tests able to be ran if cognition sys is online
#### - etc

## 3. add functionality to interact with APIs

## 4. add functionality to visualize and analyze data (graphs, charts, etc) (diagnostics)

## 5. add functionality to export data (CSV, JSON, etc) (diagnostics)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Particle-Based Cognition Engine - Main Window")
        self.setGeometry(100, 100, 800, 600)
        
        self.main_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.main_widget)

        self.layout = QtWidgets.QVBoxLayout()
        self.main_widget.setLayout(self.layout)


