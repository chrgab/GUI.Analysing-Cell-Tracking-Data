
from PyQt5.QtCore import Qt, QSize, QObject, pyqtSignal, QRunnable, pyqtSlot, QThreadPool
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QLabel,
    QLineEdit,
    QMainWindow,
    QSpinBox,
    QWidget, QDialog,
    QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QScrollArea, QProgressBar, QFrame, QMessageBox, QDoubleSpinBox
)
import traceback, sys



class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    error
        tuple (exctype, value, traceback.format_exc() )
    progress
        int indicating % progress

    '''
    error = pyqtSignal(tuple)
    progress = pyqtSignal(int)
    result = pyqtSignal(object)

class Worker(QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress
        self.kwargs['result_callback'] = self.signals.result

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''

        # Retrieve args/kwargs here; and fire processing using them
        try:
            self.fn(*self.args, **self.kwargs)
        except Exception:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))


class ScrollLabel(QScrollArea):

    def __init__(self, *args, **kwargs):
        QScrollArea.__init__(self, *args, **kwargs)
        # making widget resizable
        self.setWidgetResizable(True)
        # making qwidget object
        content = QWidget(self)
        self.setWidget(content)

        # vertical box layout
        lay = QVBoxLayout(content)

        # creating label
        self.label = QLabel(content)

        # setting alignment to the text
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        # making label multi-line
        self.label.setWordWrap(True)

        # adding label to the layout
        lay.addWidget(self.label)

    # the setText method
    def add_text(self, text):
        if not self.label.text():
            self.label.setText(" ")
        self.label.setText(self.label.text() + text)

    def set_text(self, text):
        # setting text to the label
        self.label.setText(text)
    def text(self):
        return self.label.text()

class ProgressWindow(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Processing.......")
        self.setMinimumSize(QSize(1800, 300))
        self.layout=QHBoxLayout()
        boxlayout = QVBoxLayout()
        print("layout generated")

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(1)
        self.progress_bar.setFixedSize(300,30)

        self.scroll_label= ScrollLabel()
        self.scroll_label.setFixedSize(350,300)
        #self.scroll_label.setSizeAdjustPolicy(QSizePolicy::Fixed)
        self.scroll_label.set_text("")

        boxlayout.addWidget(self.scroll_label)
        boxlayout.addWidget(self.progress_bar)

        #layout.addWidget(self.scroll_widget)
        #print("progress_bar generated")

        stop_button = QPushButton("Close")
        stop_button.setFixedSize(80,30)
        stop_button.clicked.connect(self.stop_process)

        #print("button generated")
        boxlayout.addWidget(stop_button)
        boxlayout.totalMaximumSize()
        self.layout.addLayout(boxlayout)

        # #figure widget

        self.figure_box_layout = QVBoxLayout()

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.figure_widget = QFrame()
        self.figure_widget_layout = QVBoxLayout(self.figure_widget)

        self.scroll_area.setWidget(self.figure_widget)

        self.figure_box_layout.addWidget(self.scroll_area)

        self.layout.addLayout(self.figure_box_layout)

        self.setLayout(self.layout)

    def set_fig(self, fig_path):
        figure = QLabel("")
        pixmap = QPixmap(fig_path)
        figure.setPixmap(pixmap)
        self.figure_widget_layout.addWidget(figure)


    def stop_process(self):
        self.close()

class AdvancedSettingsWindow(QDialog):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window

        self.setWindowTitle('Advanced Settings')
        self.layout = QVBoxLayout()
        # file suffix
        self.file_suffix_layout = QHBoxLayout()
        self.file_suffix_label = QLabel("Dataset Suffix")
        self.input_file_suffix = QLineEdit(self.main_window.advanced_settings["suffix"])
        self.file_suffix_layout.addWidget(self.file_suffix_label)
        self.file_suffix_layout.addWidget(self.input_file_suffix)
        self.layout.addLayout(self.file_suffix_layout)

        # threshold for size_jumps
        self.sj_threshold_layout = QHBoxLayout()
        self.label_size_jump_threshold = QLabel('Threshold size jumps')


        self.input_size_jump_threshold = QDoubleSpinBox(self)
        self.input_size_jump_threshold.setValue(
            self.main_window.advanced_settings["size_jump_threshold"])
        self.input_size_jump_threshold.setSingleStep(0.01)
        self.input_size_jump_threshold.setDecimals(2)

        self.sj_threshold_layout.addWidget(self.label_size_jump_threshold)
        self.sj_threshold_layout.addWidget(self.input_size_jump_threshold)

        self.layout.addLayout(self.sj_threshold_layout)

        # threshold for tracking_marker_division_peak
        self.tmdp_threshold_layout = QHBoxLayout()
        self.label_tracking_marker_division_peak_threshold = (
            QLabel('Threshold tracking marker division peak'))


        self.input_tracking_marker_division_peak_threshold = QDoubleSpinBox(self)
        self.input_tracking_marker_division_peak_threshold.setValue(
            self.main_window.advanced_settings["tracking_marker_division_peak_threshold"])
        self.input_tracking_marker_division_peak_threshold.setSingleStep(0.05)
        self.input_tracking_marker_division_peak_threshold.setMinimum(-9.95)
        self.input_tracking_marker_division_peak_threshold.setDecimals(2)

        self.tmdp_threshold_layout.addWidget(self.label_tracking_marker_division_peak_threshold)
        self.tmdp_threshold_layout.addWidget(self.input_tracking_marker_division_peak_threshold)

        self.layout.addLayout(self.tmdp_threshold_layout)


        #threshold for tracking_marker_jumps
        self.tmj_threshold_layout = QHBoxLayout()

        self.label_tracking_marker_jump_threshold = QLabel('Threshold tracking marker jumps')
        self.input_tracking_marker_jump_threshold = QDoubleSpinBox(self)
        self.input_tracking_marker_jump_threshold.setValue(
            self.main_window.advanced_settings["tracking_marker_jump_threshold"])
        self.input_tracking_marker_jump_threshold.setSingleStep(0.01)
        self.input_tracking_marker_jump_threshold.setDecimals(2)

        self.tmj_threshold_layout.addWidget(self.label_tracking_marker_jump_threshold)
        self.tmj_threshold_layout.addWidget(self.input_tracking_marker_jump_threshold)
        self.layout.addLayout(self.tmj_threshold_layout)

        self.button_layout = QHBoxLayout()
        self.submit_button = QPushButton('Apply', self)
        self.submit_button.clicked.connect(self.apply_settings)
        self.cancel_button = QPushButton('Cancel', self)
        self.cancel_button.clicked.connect(self.cancel)
        self.button_layout.addWidget(self.submit_button)
        self.button_layout.addWidget(self.cancel_button)
        self.layout.addLayout(self.button_layout)
        self.setLayout(self.layout)

    def cancel(self):
        self.close()
    def apply_settings(self):
        setting_update = {
            "size_jump_threshold":
                self.input_size_jump_threshold.value(),
            "tracking_marker_jump_threshold" :
                self.input_tracking_marker_jump_threshold.value(),
            "tracking_marker_division_peak_threshold" :
                self.input_tracking_marker_division_peak_threshold.value(),
            "suffix": self.input_file_suffix.text(),
        }
        self.main_window.change_advanced_settings(setting_update)
        self.main_window.search_input_folder()
        self.close()


class HelpWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Advanced Settings')
        self.layout=QVBoxLayout()
        self.scroll_label = ScrollLabel()
        self.scroll_label.setFixedSize(500, 300)
        self.layout.addWidget(self.scroll_label)

        self.setLayout(self.layout)

        text = """
        Settings:
        
        Main Folder
            Choose a folder which contains the tm-output.csv files
            (suffix can be changed in advanced settings)
          
        Digits and delimiter
            The PostProcessor can combine data from several output_files,
            if these have the same name base name, followed by a separator
            and counting numbers. Note that all numbers have the 
            same amount of digits (i.e. start with 001 for three digits).
            
            In case of problems, double-check all file name formats, 
            rename or move files whose names do not follow the pattern,
            delete delimiter, or set digits to 0        
            
            If digits is set to 0, all files are processed individually.
             
        Channels
            Provide number of channels present in the data. Must not 
            exceed number of channels in the data. The PostProcessor will
            start to extract data from Channel 1 until number of channels
            given is reached. 
            
        Tracking Channel
            Provide the number of the channel containing the tracking 
            marker.
            
        Channel Names
            You can provide names for the channels (e.g. YFP, GFP_bAct).
            These names will appear in the output Excel files and Figures.
            Do not use special characters other than underscores (i.e.:_).
            
            Renaming Channels does not affect #Channels to be used.
            
        Minimum Timepoints Per Track
            The minimum of consecutive timepoints that must contain to
            a to appear in the output. Higher numbers also speed up 
            processing.
            
        ADVANCED SETTINGS
        
        Default advanced settings are optimized for using U-2 OS cells,
        a H2B-coupled traking marker, and and imaging interval of 1h.
        May have to be adjusted for other experimental setups.  
        
        Threshold size jumps
            Minimum relative change that is defined as jump in nuclear size
            
            Sudden jumps in nuclear size usually occur due to one of 
            three reasons:
            (1) the cell undergoes division
            (2) the cell moves moves into or out of the imaging frame
            (3) segmentation or tracking errors
            PostProcessor detects size jumps and flags timepoints if
            not related to division (tracking marker peak together or 
            followed by size jump, see below).
            
            High values will result in less stringent quality control,
            whereas too low values may sort out tracks from nuclei which 
            undergo normal fluctuations in nucleus area.
            
            May be lowered for shorter imaging intervals.
            
            Can be set to very high values to disable Qualitiy Control
            based on cell size.
            
        Threshold tracking marker division peak
            
            PostProcessor detects peaks in tracking marker intensity to 
            detect cell divisions. This is due to the observation that 
            histone-coupled fluorescence markers condense in bright spots
            during cell division together with the chromatides, resulting in 
            apparent small nucleus size and a peak of average intensity. Peaks are 
            detected by:
                (1) subtracting a running-avarage from 7 timepoints to eliminate
                    trends
                (2) detecting peaks as values that are greater than 
            standard deviation multiplied by the given threshold.
            
            If value is negative, detects troughs in tracking marker average 
            intensity, which may be useful for non-histone based nuclear markers.
            
            May be adjusted for different imaging intervals.
            
        Threshold tracking marker jumps
        
            Large jumps in average tracking marker intensity can be due:
                (1) the cell undergoes division (see above)
                (2) artifacts/cell debris overlaying the nucelus
                (3) segmentation or tracking errors
            PostProcessor detects these jumps and flags timepoints if
            not related (+/- 2 hours to division.   
            
            High values will result in less stringent quality control,
            whereas too low values may sort tracks from nuclei which undergo
            normal fluctuations in tracking marker intensity.   
            
            Can be set to very high values to disable Qualitiy Control
            based on tracking marker intensity.
        
  
          
          
        """

        self.scroll_label.set_text(text)