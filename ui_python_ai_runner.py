# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'pjn_ui.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!


import sys
import numpy as np
import pandas as pd
import time

from ai_runner.stm_ai_runner import AiRunner

from PyQt5.QtWidgets import QPushButton, QHeaderView, QErrorMessage, QSizePolicy, QAbstractItemView, QVBoxLayout, QHBoxLayout, QWidget, QGridLayout, QTableWidget, QTableWidgetItem, QMainWindow, QProgressBar, QAction, QComboBox, QMessageBox, QApplication, QStyleFactory, QFrame, QLabel, QComboBox, QFileDialog, QTextEdit
from PyQt5.QtWidgets import QMenu, QToolButton, QMenuBar
from PyQt5.QtGui import QPixmap, QFont, QColor, QIcon, QPainter, QPen
from PyQt5.QtCore import QSize, Qt, QRect
from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
from time import sleep
from PIL import Image
import argparse


__author__ = "Pau Danilo Email: danilo.pau@st.com, Carra Alessandro"
__copyright__ = "Copyright (c) 2018, STMicroelectronics"
__license__ = "CC BY-NC-SA 3.0 IT - https://creativecommons.org/licenses/by-nc-sa/3.0/"


parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
args = parser.parse_args()

def createList(r1, r2): 
    return np.arange(r1, r2+1, 1)

col_val=tuple(createList(0,3071))

class Converter:
    def _map_to_fixed_point(self, values, fmt):
        """
        Map a list of arrays to fixed point by rounding (+0.5 if positive, -0.5
        otherwise). Optionally add bias to the maximum value to map asymetric
        ranges.
        """
        
        assert isinstance(fmt, tuple) and len(fmt) == 4
        min_value, max_value, scale, zero = fmt

        # Map values to fixed point representation
        out = np.round(np.asarray(values) * scale).astype(np.int64) - zero

        # clip values to avoid wrap-around problems
        out = np.clip(out, min_value, max_value)

        return out

    def int_format_quantize(self, values, fmt, signed=False):
        """Map a list of arrays to an integer format with a number of bits given
        by the format specification and using the given scale and zero point."""
        
        bits, scales, zeros = fmt

        if signed:
            min_value, max_value = -2**(bits - 1), 2**(bits - 1) - 1
        else:
            min_value, max_value = 0, 2**bits - 1

        def _quantize(values, scale, zero):
            assert scale > 0.0

            fmt = (min_value, max_value, 1.0 / scale, -zero)
            return self._map_to_fixed_point(values, fmt)

        return _quantize(values, scales, zeros)
        
    def dequantize(self, values, scale, zero=0):
        """Maps an array of quantized values to floating point using the given zero
        point and scale."""
        
        return (values.astype(np.float32) - zero) * scale

    def from_float(self, inputs, desc):
    
      if desc['type'] == np.uint8 and desc['scale'] != 0:
          outs =  self.int_format_quantize(inputs, (8, desc['scale'], desc['zero_point']))
          return outs.astype(np.uint8) 
      
      elif desc['type'] == np.int8 and desc['scale'] != 0:
          outs =  self.int_format_quantize(inputs, (8, desc['scale'], desc['zero_point']), True)
          return outs.astype(np.int8) 
             
      else: # float type
          return inputs.astype(np.float32)

    def to_float(self, outputs, desc):
    
      if desc['type'] != np.float32 and desc['scale'] != 0:
          outs =  self.dequantize(outputs, desc['scale'], desc['zero_point'])
          return outs.astype(np.float32) 

      else:
          return outputs.astype(np.float32) 

class Window(QMainWindow):
    networks = []
    j=0
    def setupUi(self, Form):
        """Instance grapichal object and discover networks and video inputs"""

        
        Form.setObjectName("Saline bottle image classification")
        Form.resize(600, 627)

        ##############################
        self.label = QtWidgets.QLabel(Form) #label = STM32.AI logo
        self.label.setGeometry(QtCore.QRect(320, 110, 260, 140))
        self.label.setStyleSheet("image: url(lib/pict2.png);")
        self.label.setText("")
        self.label.setObjectName("label")
        ##############################
        self.label_2 = QtWidgets.QLabel(Form) #label_2 = Subtitle
        self.label_2.setGeometry(QtCore.QRect(75, 30, 450, 45))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(11)
        font.setBold(True)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        ##############################
        self.label_4 = QtWidgets.QLabel(Form) #label_4 = ST Life Augmented logo
        self.label_4.setGeometry(QtCore.QRect(20, 110, 260, 140))
        #self.label_4.setStyleSheet("image: url(lib/ST_logo_2020_blue_V.jpeg);")     
        self.label_4.setStyleSheet("image: url(lib/ST_logo_2020_blue_V_sfondo.png);")
        self.label_4.setText("")
        self.label_4.setObjectName("label_4")
        ##############################
        self.label_6 = QtWidgets.QLabel(Form) #label_6 = "Danilo Pau"
        #self.label_6.setGeometry(QtCore.QRect(60, 370, 221, 41))
        self.label_6.setGeometry(QtCore.QRect(200, 250, 200, 100))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(12)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        ##############################
        self.progressBar = QtWidgets.QProgressBar(Form) #progressBar = display number of image processed
        self.progressBar.setGeometry(QtCore.QRect(210, 530, 200, 23))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        ##############################
        self.pushButton = QtWidgets.QPushButton(Form) #pushButton_1 = Select validation file button
        self.pushButton.setGeometry(QtCore.QRect(210, 482, 180, 31))
        self.pushButton.setObjectName("pushButton")
        ##############################
        self.pushButton_3 = QtWidgets.QPushButton(Form) #pushButton_3 = Select label file button
        self.pushButton_3.setGeometry(QtCore.QRect(210, 398, 180, 31))
        self.pushButton_3.setObjectName("pushButton_3")
        ##############################
        self.pushButton_4 = QtWidgets.QPushButton(Form) #pushButton_4 = Select image button
        self.pushButton_4.setGeometry(QtCore.QRect(210, 440, 180, 31))
        self.pushButton_4.setObjectName("pushButton_4")
        ##############################
        self.pushButton_5 = QtWidgets.QPushButton(Form) #pushButton_5 = Refresh network and camera button
        self.pushButton_5.setGeometry(QtCore.QRect(159, 377, 31, 31))
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_5.setIcon(QIcon("lib/gui_refresh_icon.png"))
        ##############################
        self.label_10 = QtWidgets.QLabel(Form) #label_10 = "refresh"
        #self.label_6.setGeometry(QtCore.QRect(60, 370, 221, 41))
        self.label_10.setGeometry(QtCore.QRect(10, 377, 139, 31))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(8)
        #self.label_10.setFont(font)
        ##############################
        #baudrate text and combobox
        self.label_11 = QtWidgets.QLabel(Form) #label_11 = "Baudrate"
        self.label_11.setGeometry(QtCore.QRect(10, 419, 100, 31))
        ##############################
        self.comboBox_3 = QtWidgets.QComboBox(Form)#comboBox_3 = dropdown list of baudate
        self.comboBox_3.setGeometry(QtCore.QRect(110, 419, 80, 31))
        self.comboBox_3.setObjectName("comboBox_3")
        self.comboBox_3.addItem("115200")
        self.comboBox_3.addItem("921600")
        ##############################
        #resoluition text and combobox
        self.label_12 = QtWidgets.QLabel(Form) #label_12 = "resolution"
        self.label_12.setGeometry(QtCore.QRect(410, 419, 100, 31))
        ##############################
        self.comboBox_4 = QtWidgets.QComboBox(Form)#comboBox_4 = dropdown list of resolutions
        self.comboBox_4.setGeometry(QtCore.QRect(510, 419, 80, 31))
        self.comboBox_4.setObjectName("comboBox_4")
        self.comboBox_4.addItem("Default")
        self.comboBox_4.addItem("NN")
        self.comboBox_4.addItem("800x600")
        self.comboBox_4.addItem("640x480")
        self.comboBox_4.addItem("320x240")
        self.comboBox_4.addItem("160x120")
        ##############################

        self.comboBox = QtWidgets.QComboBox(Form)#comboBox = dropdown list of network discovered
        self.comboBox.setGeometry(QtCore.QRect(210, 356, 180, 31))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.activated.connect(self.check_combobox_network)
        ##############################
        #camera number text and combobox
        self.label_13 = QtWidgets.QLabel(Form) #label_13 = "Camera n°"
        self.label_13.setGeometry(QtCore.QRect(410, 377, 100, 31))
        ##############################
        self.comboBox_2 = QtWidgets.QComboBox(Form)#comboBox_2 = dropdown list of video input discovered
        self.comboBox_2.setGeometry(QtCore.QRect(510, 377, 80, 31))
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.addItem("")
        self.comboBox_2.activated.connect(self.check_combobox_camera)
        # self.checkBox = QtWidgets.QCheckBox(Form)
        # self.checkBox.setGeometry(QtCore.QRect(450, 450, 70, 17))
        # self.checkBox.setObjectName("checkBox")
        ##############################
        self.pushButton_2 = QtWidgets.QPushButton(Form) #pushButton_2 = Camera button button
        self.pushButton_2.setGeometry(QtCore.QRect(410, 461, 180, 31))
        #self.pushButton_2.setStyleSheet("image: url(lib/gaia_logo1.png);")
        self.pushButton_2.setObjectName("pushButton_2")
        ##############################



        #end layout and widget
        
        self.retranslateUi(Form)

        self.pushButton.clicked.connect(self.file_csv_open)
        self.pushButton_2.clicked.connect(self.live_cam)
        self.pushButton_3.clicked.connect(self.file_label_open) #apertura file label 
        self.pushButton_4.clicked.connect(self.image_open) #apertura immagine 
        self.pushButton_5.clicked.connect(self.network_discovery) #refresh network
        
        self.file_type_old = ''
        self.first_discovery = True
        self.button_disabled()
        self.network_discovery()
        self.counter = 0
        self.res_win = Results(self)         
        
        #self.progressBar.valueChanged['int'].connect(Form.progress_bar)
        QtCore.QMetaObject.connectSlotsByName(Form)

    
    def retranslateUi(self, Form):
        """SetText of the graphical object"""
        
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "ST Tutorial on Tiny Neural Network"))
        
        self.label_2.setText(_translate("Form", "ST Tutorial on Tiny Neural Network\n"
        "with STM32Cube.AI on STM32 Microcontroller"))
        self.label_2.setAlignment(Qt.AlignCenter)
        
        self.label_6.setText(_translate("Form", "Danilo Pau\n Alessandro Carra\n V2.1.1"))
        self.label_6.setAlignment(Qt.AlignCenter)
        
        self.label_10.setText(_translate("Form", "Refresh\nNN and camera"))
        self.label_10.setAlignment(Qt.AlignCenter)

        self.label_11.setText(_translate("Form", "Baudrate"))
        self.label_11.setAlignment(Qt.AlignCenter)
        
        self.label_12.setText(_translate("Form", "Resolution"))
        self.label_12.setAlignment(Qt.AlignCenter)

        self.label_13.setText(_translate("Form", "Camera n°"))
        self.label_13.setAlignment(Qt.AlignCenter)
    
        self.comboBox_2.setItemText(0, _translate("Form", "Select Webcam"))     
        self.pushButton.setText(_translate("Form", "Select Validation File"))
        self.pushButton_2.setText(_translate("Form", "GO"))
        self.pushButton_3.setText(_translate("Form", "Select label file"))
        self.pushButton_4.setText(_translate("Form", "Select image"))

           
    def progressBar(self):
        """Set the percentage of the progress bar"""
        
        progressBar = QProgressBar(self)
        progressBar.setRange(0,100)
        """
        if progressBar.value == 100 : 
            progressBar.setVisible(False)
        elif progressBar.value == 0: 
            progressBar.setVisible(False)
        else :
            progressBar.setVisible(True)
        """
        return progressBar

    

                
    def openResults(self):
        """ Instance Result window object and open it"""
    
        if self.file_type != self.file_type_old or self.file_type == 'csv' or not self.res_win.isVisible():
            self.res_win.clean_data()
            self.res_win.init_labels()
            if not self.res_win.isVisible():
                self.dialogs.append(self.res_win)
                self.res_win.show()
                
        if self.file_type == 'csv':     
            self.res_win.setup_v()

        elif self.file_type == 'image':
            self.res_win.setup_t()

        self.file_type_old = self.file_type


    def convert (self, x):
        """ Convert input value to one label if it matches"""
        
        if np.size(x)==1:
            try:
                outputs = self.labels[np.float32(x)]
            except(KeyError):
                outputs = 'no_class'

        return str(outputs)

        
    def convert_outputs1 (self, x):
        """ Convert output values to the corresponding label if they match"""
        outputs = []
        for i in range(0,len(x)):
            try:
                outputs.append(self.labels[np.float32(x[i][0])])
            except(KeyError):
                outputs.append('no_class')

                if args.verbose:  print('do not be able to determine the class' + '\n' + 'sample n° ' + str(i) + '\n' + str(x[i])+ '\n')        
        return outputs

    def file_open(self, filepath):
        """ Open the *.csv file (filepath) that contain input data for the validation.
        Check and create a key,value dict if it has not been already created. 
        Check if the data in the selected file have the correct dimension."""
                    
        if filepath != '':
            if args.verbose:  print("Opening file      : ", filepath, flush=True)
            if args.verbose:  print("Selected c-name   : ", self.c_name, flush=True)

            t_input_desc = self.nn.get_input_infos(name=self.c_name)[0]
            t_output_desc = self.nn.get_output_infos(name=self.c_name)[0]
 
            i_shape = t_input_desc['shape'][1:]
            i_dtype = t_input_desc['type']
            o_shape = t_output_desc['shape'][1:]           
            
            try:
                isinstance(self.labels, dict)
            except:
                self.labels ={}
                for i in range(0, o_shape[2]):
                    self.labels[np.float32(i)] = str(i)
            
            type(self).labels = self.labels.copy()
            
            inputs = np.genfromtxt(filepath, delimiter=',')
                        
            if inputs.size == inputs.shape[0] :
                    inputs = inputs.reshape((1,inputs.shape[0]))

            class_label = [ self.convert(_c) for _c in inputs[:,-1]]
            inputs = inputs[:, :-1]

            if inputs.shape[1] == i_shape[0] * i_shape[1] * i_shape[2]:
                converter = Converter()

                outputs_temp = []
            
                CPU_duration = 0
                start_USB_duration = time.time()
                
                type(self).classes = class_label

                number_samples = inputs.shape[0]
                adur = 0.0
                
                for i in range(number_samples):
                
                    in_values = converter.from_float(inputs[i], t_input_desc)

                    in_values = in_values.reshape((1,) + i_shape)
                    assert i_dtype == in_values.dtype
                    in_values = np.ascontiguousarray(in_values.astype(i_dtype))

                    # start_CPU_duration = time.time()
                    out_values, profile =  self.nn.invoke(in_values, name=self.c_name)

                    adur += np.mean(profile['c_durations'])
                    
                    # end_CPU_duration = time.time()
                    
                    out_values = converter.to_float(out_values[0], t_output_desc)
                    CPU_duration += profile['debug']['host_duration'] # end_CPU_duration-start_CPU_duration
                    self.progressBar.setValue(int(i*100/number_samples))

                    outputs_temp.append(np.argmax(out_values, axis=-1).flatten())

                type(self).outputs = self.convert_outputs1(outputs_temp)            

                self.progressBar.reset()
                self.progressBar.setValue(0)
                end_USB_duration = time.time()
                USB_duration = end_USB_duration - start_USB_duration

                type(self).USB_rate = int(number_samples/USB_duration)
                type(self).CPU_rate = int(number_samples/CPU_duration)
                type(self).inference_time = adur/number_samples
                type(self).device_desc = self.nn.get_info()['device']['desc']
                
                self.dialogs = list()
                self.openResults()
    
            else : 
                self.display_error("Wrong file size")
                
    def network_discovery(self):
        """ Open the connection and search for networks and add networks names to the combobox dropdown list"""
        c_baudrate = int(self.comboBox_3.currentText())
        
        if not self.first_discovery:
            if self.nn:
              self.nn.disconnect()
            if args.verbose: print('Closing connection ...', flush=True)
        self.first_discovery = False
        if args.verbose: print('Opening connection ...', flush=True)
        
        self.nn = AiRunner()
        con = self.nn.connect('serial', baudrate=c_baudrate)  # auto-detect mode
        print('Connection: ', con)

        if self.nn.is_connected and self.nn.names:
            print(self.nn)
            print(self.nn.get_info()['device']['desc'])
            print('Network(s) found = {}'.format(self.nn.names))
            for c_nn in self.nn.names:
                info_nn = self.nn.get_info(name=c_nn)
                s_ins = [str(s_['shape'][1:]) + ':' + str(np.dtype(s_['type'])) for s_ in info_nn['inputs']]
                s_outs = [str(s_['shape'][1:]) + ':' + str(np.dtype(s_['type'])) for s_ in info_nn['outputs']]
                print(' {} : {} -> {} -> {} macc={} rom={:.2f}KiB ram={:.2f}KiB'.format(info_nn['name'],
                                                    ('/').join(s_ins),
                                                    info_nn['n_nodes'] ,
                                                    ('/').join(s_outs),
                                                    info_nn['macc'],
                                                    info_nn['weights'] / 1024,
                                                    info_nn['activations'] / 1024
                                                    ),
                      flush=True) 
                if args.verbose: print(info_nn, flush=True)

            self.comboBox.clear()
            self.comboBox.addItem("Select Network") 

            for i in self.nn.names:  # networks:
                self.comboBox.addItem(i)
        else:
            self.nn = None
            print('no network', flush=True)
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("No network found")
            msg.setWindowTitle("Error")
            msg.exec_()
            
        self.check_combobox_network()
        self.webcam_discovery()
        
    def webcam_discovery(self):
        """ Search for video inputs and add items to the combobox dropdown list """
        
        self.comboBox_2.clear()
        self.comboBox_2.addItem("Camera")
           
           
        for i in range(5):
            if sys.platform == 'linux':
                webcam = cv2.VideoCapture(i,cv2.CAP_V4L)
                if args.verbose: print("CAP_V4L")
            elif sys.platform == 'win32':
                webcam = cv2.VideoCapture(i,cv2.CAP_DSHOW)
                if args.verbose: print("CAP_DSHOW")
            else:
                webcam = cv2.VideoCapture(i)   
            if webcam != None and  webcam.isOpened():
                self.comboBox_2.addItem(str(i)) 

            webcam.release()
            cv2.destroyAllWindows()
            
 
    def check_combobox_network(self):
        """ Check the selected item of the network combobox and if necessary display error"""
        
        self.webcam_discovery
        self.c_name = self.comboBox.currentText()
        if self.c_name != "Select Network":
            if self.nn and self.c_name not in self.nn.names:
                if self.c_name != "Select Network":
                    self.display_error("Select a network!")
                    self.button_disabled()
                    return False
            else:
                self.check_combobox_camera()
                self.button_enabled()
                return True
        else:
            self.button_disabled()


    def check_combobox_camera(self):
        """ Check the selected item of the camera combobox and if necessary display error"""
        
        if self.comboBox_2.currentText() != "Camera":
            self.pushButton_2.setEnabled(True)      
        else:
            self.pushButton_2.setEnabled(False)      

        
    def button_enabled(self):
        """ Enable the button of the GUI"""
        
        self.pushButton.setEnabled(True)        
        #self.pushButton_2.setEnabled(True)        
        self.pushButton_3.setEnabled(True)        
        self.pushButton_4.setEnabled(True)  
        
    def button_disabled(self):
        """ Disable the button of the GUI"""
        
        self.pushButton.setEnabled(False)        
        self.pushButton_2.setEnabled(False)        
        self.pushButton_3.setEnabled(False)        
        self.pushButton_4.setEnabled(False)        

    
    def file_csv_open(self):  
        """ Open *.csv file selected from the GUI"""
        
        name = QFileDialog.getOpenFileName(self, 'Open File', filter="CSV files (*.csv)")[0]
        self.file_type = 'csv'
        self.file_open(name)   
        
    def file_label_open(self):
        """ Open key,value labels file and check if they are equal to the number of the classes"""
        
        self.l_name = QFileDialog.getOpenFileName(self, 'Open File')[0]
        label = {}
        
        # t_out = self.nn.output_tensor(name=self.c_name)
        out_shape = self.nn.get_output_infos(name=self.c_name)[0]['shape'][1:]
     
        if self.l_name != '':
            try :
                if args.verbose:  print("Opening: " +str(self.l_name))
                with open(self.l_name) as myfile:
                    for line in myfile:
                        key, value = line.partition(",")[0::2]
                        label[np.float32(key)] = value.rstrip()

                if args.verbose: print('Label da file:',label)

                if len(label) != out_shape[2]:
                    self.display_error('wrong number of labels!')

                else:
                    self.labels = label
                    self.res_win = Results(self)

            except:
                self.display_error('wrong file label!')

        """        
        else:
            for i in range(0,out_tens.shape[2]):
                label[np.float32(i)] = str(i)
        """

    def display_error(self,error_message='General Error'):
        """ Display an error with 'error_message' text"""
        
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText(error_message)
        msg.setWindowTitle("Error")
        msg.exec_()


    def live_cam(self):
        """ Open video stream and capture an image or a sequence of images, according to the pressed button 
        Press S: validate one image
        Press L: validate a sequence of images
        Press P: pause loop
        Press Q: quit"""
        
        if not self.check_combobox_network():
            self.display_error('A network should be selected ...')
        else :
            self.pushButton_2.setEnabled(False)    #disable camera button  

            w_name = self.comboBox_2.currentText()


            if sys.platform == 'linux':
                webcam = cv2.VideoCapture(int(w_name),cv2.CAP_V4L)
            elif sys.platform == 'win32':
                webcam = cv2.VideoCapture(int(w_name),cv2.CAP_DSHOW)
            else:
                webcam = cv2.VideoCapture(int(w_name))   
            
            webcam.set(cv2.CAP_PROP_BUFFERSIZE, 3)

            # in_tens = self.nn.input_tensor(name=self.c_name)
            in_shape = self.nn.get_input_infos(name=self.c_name)[0]['shape'][1:]

            w_res = self.comboBox_4.currentText()
            if w_res =="Default":
                webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            elif w_res =="NN":
                webcam.set(cv2.CAP_PROP_FRAME_WIDTH, in_shape[0])
                webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, in_shape[1])
            elif w_res =="800x600":
                webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
                webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
            elif w_res =="640x480":
                webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            elif w_res =="320x240":
                webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            elif w_res =="160x120":
                webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
                webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
                           
            #check if webcam is opened or busy
            if not webcam.isOpened():
                self.display_error('Webcam busy!! close it and retry')
                return  
            
            
            flag_loop=False
            
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText('S validate one image\nL validate in loop\nP pause loop\nQ quit')
            msg.setWindowTitle("Capturing")
            msg.exec_()
    
            while True:
                try:
                    check, frame = webcam.read()
                    
                    if not check:
                        self.display_error('Webcam error, retry')
                        break
                    cv2.imshow("Capturing", frame)
                    key = cv2.waitKey(1)

                    if key == ord('s'):
                        if args.verbose: print("Saving image from webcam")
                        cv2.imwrite(filename='temp_cv.jpg', img=frame)
                        self.process_image(file_name='temp_cv.jpg')
                        flag_loop=False                        

                    elif key == ord('q') or cv2.getWindowProperty("Capturing", cv2.WND_PROP_VISIBLE) <= 0:
                        webcam.release()
                        cv2.destroyAllWindows()
                        flag_loop=False
                        self.pushButton_2.setEnabled(True)    #enable camera button  
                        break
                        
                    elif key == ord('p') and flag_loop==True:
                        flag_loop = False
                        
                    elif key == ord('l') or flag_loop==True:
                        if args.verbose:  print("Loop from webcam")
                        cv2.imwrite(filename='temp_cv.jpg', img=frame)
                        self.process_image(file_name='temp_cv.jpg')
                        flag_loop = True

                except(KeyboardInterrupt):
                    if args.verbose:  print("Turning off camera.")
                    webcam.release()
                    if args.verbose:  print("Camera off.")
                    if args.verbose:  print("Program ended.")
                    cv2.destroyAllWindows()
                    self.pushButton_2.setEnabled(True)    #enable camera button  

                    break
                    



    def process_image(self, file_name):
        """ Process image captured from the camera or selected from the GUI.
        Open->resize->reshape->flatten->save .csv->self.open_file()"""
        
        # in_tens = self.nn.input_tensor(name=self.c_name)
        in_shape = self.nn.get_input_infos(name=self.c_name)[0]['shape'][1:]
            
        #ottimizzato
        if in_shape[2]==1:
            img_ = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)

        elif in_shape[2]==3:
            img_ = cv2.imread(file_name, cv2.IMREAD_COLOR)
            img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)

        else:
            img_ = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
        
        img_ = cv2.resize(img_,(in_shape[0],in_shape[1]))
        cv2.imwrite(filename='resized_temp.jpg', img=img_)
        image_array = np.asarray(img_)
        image_array = image_array.reshape((1,image_array.shape[0]*image_array.shape[1]*in_shape[2],1))
        output = 0
        image_array = np.append(image_array,output)
        image_array = image_array.reshape((1,image_array.shape[0]))
        image_array = image_array/255

        with open("temp_val.csv", 'wb') as f:
            np.savetxt(f,image_array,newline='\n' , delimiter=",")#Saving the list as a csv    
        self.file_type = "image"
        self.file_open("temp_val.csv")

        
    def image_open(self):
        """Open the selected image and check if is compatible with opencv"""
        
        name = QFileDialog.getOpenFileName(self, 'Open Image')[0]
        if name !='':
            if np.any(cv2.imread(name, cv2.IMREAD_UNCHANGED)) == None:
                self.display_error('Immagine non leggibile! Riprovare')
            else:
                if args.verbose:print("Opening: " +str(name))
                self.process_image(file_name=name)

        

    def editor(self):
        self.textEdit = QTextEdit()
        self.setCentralWidget(self.textEdit)

class Results(QMainWindow):
    
    accuracy_ratio = 0
    intrusion_ratio = 0
    classification = {} #classification = {'level_100':0, 'level_50':0, 'level_80':0, 'level_empty':0}
    confusion_matrix = {}#confusion_matrix = {'level_100':{'level_100':0, 'level_50':0, 'level_80':0, 'level_empty':0}, ...
    error_rate = {} #error_rate = {'level_100':0, 'level_50':0, 'level_80':0, 'level_empty':0}
    
    def __init__(self, parent=None):
        super(Results,self).__init__()
        self.error_rate = {}
        self.table_string = []
        self.table_bool = []
        self.accuracy_ratio = []
        self.intrusion_ratio = 0
        self.classification = {}
        self.confusion_matrix = {}
        self.item_number = 0        
    
    
    def init_labels(self):
        labels = Window.labels.copy()
        self.labels_inv = {} #label con values : 0
        self.labels_cf = {}
        for i in labels.values():
            self.labels_inv[i] = 0
        for i in labels.values():
            self.labels_cf[i]= self.labels_inv.copy()
        self.item_number = 0

        
        self.error_rate = self.labels_inv.copy()
        self.classification = self.labels_inv.copy()
        self.confusion_matrix = self.labels_cf.copy()
    
    #def setup(self, parent=None):
    def setup_v(self):
        """ Set the Results window and initialize the variables """
        
        self.setStyleSheet("QMainWindow {background: 'white';}");
        self.setGeometry(100, 100, 750, 500)
        self.setMinimumSize(QSize(480, 80))         # Set sizes 
        self.setWindowTitle("ST demonstrator on Tiny Neural Network: Validation Results")

        self.setCentralWidget(QFrame())

        grid_layout = QGridLayout()         # Create QGridLayout

        
        grid_layout.addLayout(self.table_v(), 0, 0, -1, 1)   # Adding the table to the grid
        grid_layout.addLayout(self.classification_samples_v(), 1, 1)
        grid_layout.addLayout(self.home(), 3, 1)
        grid_layout.addLayout(self.efficiency(), 0, 1)
        grid_layout.addLayout(self.confusionMatrix(), 2, 1)
         
        self.centralWidget().setLayout(grid_layout)
        #self.clean_data()    def setup(self):
        
        
    def setup_t(self):
        """ Set the Results window and initialize the variables """
        
        self.setStyleSheet("QMainWindow {background: 'white';}");
        self.setGeometry(100, 100, 750, 500)
        self.setMinimumSize(QSize(480, 80))         # Set sizes 
        self.setWindowTitle("ST demonstrator on Tiny Neural Network: Test Results")

        self.setCentralWidget(QFrame())

        grid_layout = QGridLayout()         # Create QGridLayout

        
        grid_layout.addLayout(self.table_t(), 0, 0, -1, 1)   # Adding the table to the grid
        grid_layout.addLayout(self.classification_samples_t(), 1, 1)
        #grid_layout.addLayout(self.home(), 3, 1)
        grid_layout.addLayout(self.efficiency(), 0, 1)
        #grid_layout.addLayout(self.confusionMatrix(), 2, 1)
                
        self.centralWidget().setLayout(grid_layout)
        #self.clean_data()
        
    def clean_data(self):
        """ Clean the results after displaying it"""
        self.error_rate = {}
        self.table_string = []
        self.table_bool = []
        self.accuracy_ratio = []
        self.intrusion_ratio = 0
        self.classification = {}
        self.confusion_matrix = {}
        #self.item_number = 0
        
    def clean_res(self):
        Window.classes = []
        Window.outputs = []
        Window.USB_rate = 0
        Window.CPU_rate = 0
        Window.inference_time = 0
        Window.device_desc = ''
    
    
    def confusionMatrix(self):
        """ Set the layout for the confusion matrix """
        
        table = QTableWidget(self)
        #table.verticalHeader().setVisible(True)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setColumnCount(len(self.classification))
        table.setRowCount(len(self.classification))
                
        table_headers=[]
        for i in self.confusion_matrix:
            for j in self.confusion_matrix[i]:
                if i not in table_headers:
                    table_headers.append(i)
                if j not in table_headers:
                    table_headers.append(j)
        # Set the table headers
        table.setHorizontalHeaderLabels(table_headers)
        table.setVerticalHeaderLabels(table_headers)
        
        for i in range(len(self.confusion_matrix)):                    
            for j in range(len(self.confusion_matrix)):
                table.setItem(i, j, QTableWidgetItem(str(0)))
                
     
        for i in self.confusion_matrix:
            for j in self.confusion_matrix[i]:
                table.setItem(table_headers.index(i), table_headers.index(j), QTableWidgetItem(str(self.confusion_matrix[i][j])))
        table.resizeColumnsToContents()
        table.resizeRowsToContents()

        box_layout = QHBoxLayout(self)

        size = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        ## Left layout
        size.setHorizontalStretch(1)
        table.setSizePolicy(size)

        box_layout.addWidget(table)

        return box_layout


    def table_v(self):
        """ Set the layout for the table"""
        error_rate = self.error_rate 
        
        #example : error_rate = {'level_100':0, 'level_50':0, 'level_80':0, 'level_empty':0}
        table = QTableWidget(self)  # Create a table
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        table.setColumnCount(2)     #Set two columns

        # Set the table headers
        table.setHorizontalHeaderLabels(["Image n°", "Predicted Class"])
 
        #Set the tooltips to headings
        #table.horizontalHeaderItem(0).setToolTip("Column 1 ")
        #table.horizontalHeaderItem(1).setToolTip("Column 2 ")

        table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        # Set the alignment to the headers
        table.horizontalHeaderItem(0).setTextAlignment(Qt.AlignHCenter)
        table.horizontalHeaderItem(1).setTextAlignment(Qt.AlignHCenter)

        for i in range(0,  len(self.table_string)):
            table.insertRow(table.rowCount())
            table.setItem(i, 0, QTableWidgetItem(str(i+1)))
            table.setItem(i, 1, QTableWidgetItem(self.table_string[i]))
            table.item(i, 0).setTextAlignment(4)
            table.item(i, 1).setTextAlignment(4)
            
            if self.table_bool[i] == True:
                table.item(i, 1).setBackground(QColor(129, 212, 26)) #celeste (137, 207,240)

            elif self.table_bool[i] == False:
                table.item(i, 1).setBackground(QColor(226,103,103)) #rosso (250,0,0)
            
                
        for i in range(0, len(Window.outputs)):
            if Window.outputs[i] not in error_rate: 
                error_rate[Window.outputs[i]] = 0
            if (Window.classes[i] != Window.outputs[i]) :
                """
                if (Window.classes[i] > Window.outputs[i]) :
                    self.table_string.append(str(Window.outputs[i]) + '(truth = unknown)')
                    self.table_bool.append(False)
                else :
                """    
                self.table_string.append(str(Window.outputs[i]) + ' (truth = ' + str(Window.classes[i]) + ')')
                self.table_bool.append(False)
                table.insertRow(table.rowCount())
                table.setItem(i+ self.item_number, 0, QTableWidgetItem(str(i+ self.item_number+1)))
                table.item(i+ self.item_number, 0).setTextAlignment(4)
                table.setItem(i+ self.item_number, 1, QTableWidgetItem(self.table_string[-1]))
                table.item(i+ self.item_number, 1).setBackground(QColor(226,103,103))
                table.item(i+ self.item_number, 1).setTextAlignment(4)
                error_rate[Window.outputs[i]] += 1
                type(self).accuracy_ratio += 1
            else:
                self.table_string.append(Window.outputs[i])
                self.table_bool.append(False)
                table.insertRow(table.rowCount())
                table.setItem(i+ self.item_number, 0, QTableWidgetItem(str(i+ self.item_number+1)))
                table.item(i+ self.item_number, 0).setTextAlignment(4)
                table.setItem(i+ self.item_number, 1, QTableWidgetItem(Window.outputs[i]))
                table.item(i+ self.item_number, 1).setBackground(QColor(129,212,26))
                table.item(i+ self.item_number, 1).setTextAlignment(4)
                
                
        self.item_number += len(Window.outputs)
                
        self.error_rate = error_rate 
        
        #type(self).accuracy_ratio = self.accuracy_ratio/len(Window.classes)
        # Do the resize of the columns by content
        table.resizeColumnsToContents()
        table.resizeRowsToContents()
                
        box_layout = QHBoxLayout(self)

        size = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        ## Left layout
        size.setHorizontalStretch(1)
        table.setSizePolicy(size)
        

        box_layout.addWidget(table)

        return box_layout
    
    def table_t(self):
        """ Set the layout for the table"""
        error_rate = self.error_rate 
        
        #example : error_rate = {'level_100':0, 'level_50':0, 'level_80':0, 'level_empty':0}
        table = QTableWidget(self)  # Create a table
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        table.setColumnCount(2)     #Set two columns

        # Set the table headers
        table.setHorizontalHeaderLabels(["Image n°", "Predicted Class"])
 
        #Set the tooltips to headings
        #table.horizontalHeaderItem(0).setToolTip("Column 1 ")
        #table.horizontalHeaderItem(1).setToolTip("Column 2 ")

        table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        # Set the alignment to the headers
        table.horizontalHeaderItem(0).setTextAlignment(Qt.AlignHCenter)
        table.horizontalHeaderItem(1).setTextAlignment(Qt.AlignHCenter)

        for i in range(0,  len(self.table_string)):
            table.insertRow(table.rowCount())
            table.setItem(i, 0, QTableWidgetItem(str(i+1)))
            table.setItem(i, 1, QTableWidgetItem(self.table_string[i]))
            table.item(i, 0).setTextAlignment(4)
            table.item(i, 1).setTextAlignment(4)

            
                
        for i in range(0, len(Window.outputs)):
            if Window.outputs[i] not in error_rate: 
                error_rate[Window.outputs[i]] = 0
            
            self.table_string.append(Window.outputs[i])
            self.table_bool.append(False)
            table.insertRow(table.rowCount())
            table.setItem(i+ self.item_number, 0, QTableWidgetItem(str(i+ self.item_number+1)))            
            table.item(i+ self.item_number, 0).setTextAlignment(4)
            table.setItem(i+ self.item_number, 1, QTableWidgetItem(Window.outputs[i]))
            table.item(i+ self.item_number, 1).setTextAlignment(4)

        self.item_number += len(Window.outputs)
                
        self.error_rate = error_rate 
        
        #type(self).accuracy_ratio = self.accuracy_ratio/len(Window.classes)
        # Do the resize of the columns by content
        table.resizeColumnsToContents()
        table.resizeRowsToContents()
                
        box_layout = QHBoxLayout(self)

        size = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        # Left layout
        size.setHorizontalStretch(1)
        table.setSizePolicy(size)
        

        box_layout.addWidget(table)

        return box_layout

    def classification_samples_t(self):
        """ Set the layout for the classification samples"""
        
        classification = self.classification
        confusion_matrix = self.confusion_matrix
        error = self.error_rate.copy()
        
        box_layout = QVBoxLayout(self)
     
        for i in range (0, len(Window.outputs)) :
            if Window.outputs[i] in classification:
                classification[Window.outputs[i]] += 1
            else :
                classification[Window.outputs[i]] = 1
            
            if Window.classes[i] not in confusion_matrix:
                confusion_matrix[Window.classes[i]] = {}
                confusion_matrix[Window.classes[i]][Window.outputs[i]] = 1
            else : 
                if Window.outputs[i] not in confusion_matrix[Window.classes[i]]:
                    confusion_matrix[Window.classes[i]][Window.outputs[i]] = 1
                else : 
                    confusion_matrix[Window.classes[i]][Window.outputs[i]] += 1



        for i in self.error_rate: 
            if classification[i]>0:
                error[i] = int(100*(classification[i]-self.error_rate[i])/classification[i])
                #self.accuracy_ratio += error[i]

            else : 
                error[i] = 0


        classification_title = QLabel('Classification of the outputs', self)
        classification_title.setFont(QFont('Arial', 12, QFont.Bold))
        box_layout.addWidget(classification_title)

        for i in classification:
            label = QLabel(i + ' : ' + str(classification[i]) + ' Images', self)
            box_layout.addWidget(label)        
        
        self.classification = classification
        self.confusion_matrix = confusion_matrix
        type(self).classification = classification
        type(self).confusion_matrix = confusion_matrix
        
        return box_layout
    
    def classification_samples_v(self):
        """ Set the layout for the classification samples"""
        
        classification = self.classification
        confusion_matrix = self.confusion_matrix
        error = self.error_rate.copy()
        
        box_layout = QVBoxLayout(self)
     
        for i in range (0, len(Window.outputs)) :
            if Window.outputs[i] in classification:
                classification[Window.outputs[i]] += 1
            else :
                classification[Window.outputs[i]] = 1
            
            if Window.classes[i] not in confusion_matrix:
                confusion_matrix[Window.classes[i]] = {}
                confusion_matrix[Window.classes[i]][Window.outputs[i]] = 1
            else : 
                if Window.outputs[i] not in confusion_matrix[Window.classes[i]]:
                    confusion_matrix[Window.classes[i]][Window.outputs[i]] = 1
                else : 
                    confusion_matrix[Window.classes[i]][Window.outputs[i]] += 1



        for i in self.error_rate: 
            if classification[i]>0:
                error[i] = int(100*(classification[i]-self.error_rate[i])/classification[i])
                self.accuracy_ratio += [int(self.error_rate[i])]

            else : 
                error[i] = 0


        classification_title = QLabel('Classification of the outputs', self)
        classification_title.setFont(QFont('Arial', 12, QFont.Bold))
        box_layout.addWidget(classification_title)

        for i in classification:
            label = QLabel(i + ' : ' + str(classification[i]) + ' Images (acc: ' + str(error[i]) + ' %)', self)
            box_layout.addWidget(label)        
        
        self.classification = classification
        self.confusion_matrix = confusion_matrix
        type(self).classification = classification
        type(self).confusion_matrix = confusion_matrix
        
        return box_layout

    def efficiency(self):
        """ Set the layout for the efficency """
                
        box_layout = QVBoxLayout(self)

        timers = QLabel('Model time efficiency', self)
        timers.setFont(QFont('Arial', 12, QFont.Bold))
        
        _ldesc = Window.device_desc.split(' ')
        if _ldesc:
            _desc = _ldesc[0] + ' ' + _ldesc[1]
        else:
            _desc = 'UNKNOWN'
        

        #USB_efficiency = QLabel('Host USB rate : ' + str(Window.USB_rate) + ' packets/s', self)
        # CPU_efficiency = QLabel('MPU rate : ' + str(Window.CPU_rate) + ' packets/s', self)
        #CPU_efficiency = QLabel('MCU rate : {} packets/s ({})'.format(int(1000/Window.inference_time), _desc), self)
        # inference_time = QLabel('Inference time : ' + str(round(Window.inference_time*1000, 3)) + ' ms/packets', self)
        inference_time = QLabel('Inference time : {:.3f} ms/Image'.format(Window.inference_time), self)

        box_layout.addWidget(timers)
        #box_layout.addWidget(USB_efficiency)
        #box_layout.addWidget(CPU_efficiency)
        box_layout.addWidget(inference_time)

        return box_layout
        


    def home(self):
        """ Set the layout for the home """
        box_layout = QVBoxLayout(self)
        
        if 'normal' in self.classification: 
            intrusion_ratio = self.classification['normal']/len(Window.classes)
        else : 
            intrusion_ratio = 0      

        efficiency = QLabel('', self)
        efficiency.setFont(QFont('Arial', 12, QFont.Bold))
        #box_layout.addWidget(efficiency)

        ratio = QLabel(' : ' + str('.') + '%')
        #box_layout.addWidget(ratio)
        
        
        self.accuracy_ratio += []
        for i in range(0,len(self.accuracy_ratio)-1):
            self.accuracy_ratio[-1] += self.accuracy_ratio[i]

        #accuracy = QLabel('Overall accuracy : ' + str(int(100-self.accuracy_ratio*100)) + '%', self)
        accuracy = QLabel('Overall accuracy : ' + str(100-(int(self.accuracy_ratio[-1])/len(Window.outputs))*100) + '%', self) 
        accuracy.setFont(QFont('Arial', 12, QFont.Bold))
        box_layout.addWidget(accuracy)
        self.accuracy_ratio = []
        return box_layout
            
    def download(self):
        self.completed = 0

        while self.completed < 100:
            self.completed += 1
            self.progress.setValue(self.completed)

if __name__ == "__main__":
    pjn=0
    if args.verbose: print("preQApplication")
    app = QtWidgets.QApplication(sys.argv)
    if args.verbose: print("postQApplication")
    Form = QtWidgets.QWidget()
    ui = Window()
    #_BAUDRATE=115200
    
    ui.setupUi(Form)
    
    Form.show()
    sys.exit(app.exec_())
