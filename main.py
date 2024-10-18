import time, datetime
from idlelib.help import HelpWindow

from PyQt5.QtWidgets import (QApplication,

    QLineEdit,
    QMainWindow,
    QSpinBox,
    QWidget, QDialog,
    QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QScrollArea, QProgressBar, QFrame, QMessageBox
)
import traceback, sys
from methods import *
from classes import *



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        #break execution
        self.keep_running = True

        #multithreading
        self.threadpool = QThreadPool()
        self.setWindowTitle("TrackMate PostProcessor")
        self.setMinimumSize(QSize(500,500))
        layout = QVBoxLayout()
        # difine variables
        self.selected_folder = "Choose Input Folder"
        self.number_channels=3
        self.file_list = []
        self.dataset_list=[]

        self.settings ={}

        self.advanced_settings={
            "size_jump_threshold"  : 0.2,
            "tracking_marker_jump_threshold" : 0.18,
            "tracking_marker_division_peak_threshold": 1.5,
            "suffix": "_tm-output.csv",
        }

    # help window
        self.help_button = QPushButton("HELP?")
        self.help_button.clicked.connect(self.open_help_window)
        layout.addWidget(self.help_button)
    # choose input file
        layout_file_selection= QHBoxLayout()
        self.selected_folder_label = QLabel(F'Input Folder: {self.selected_folder}')
        choose_folder_button = QPushButton("Choose Input Folder")
        choose_folder_button.clicked.connect(self.choose_folder_button_clicked)
        choose_folder_button.setFixedSize(QSize(140, 30))
        layout_file_selection.addWidget(self.selected_folder_label)
        layout_file_selection.addWidget(choose_folder_button)
        layout.addLayout(layout_file_selection)

    # file suffix + separator
        label_specify = QLabel("Specify how replicates are labeled")
        label_specify.setAlignment(Qt.AlignHCenter | Qt.AlignBottom)
        layout.addWidget(label_specify)

        replicate_number_layout = QHBoxLayout()
        replicate_digits_label = QLabel("digits")
        replicate_digits = QSpinBox()
        replicate_digits.setMinimum(0)
        replicate_digits.setValue(2)
        separator_label = QLabel("delimiter")
        separator_input = QLineEdit("_")
        self.subset_separator = separator_input.text()
        self.subset_digits = replicate_digits.value()
        self.replica_format = self.get_replica_format()
        self.replica_format_label = QLabel(self.replica_format)
        replicate_digits.valueChanged.connect(self.change_digits)
        separator_input.textChanged.connect(self.change_seperator)
        replicate_number_layout.addWidget(replicate_digits_label)
        replicate_number_layout.addWidget(replicate_digits)
        replicate_number_layout.addWidget(separator_label)
        replicate_number_layout.addWidget(separator_input)
        replicate_number_layout.addWidget(self.replica_format_label)
        layout.addLayout(replicate_number_layout)

    #datasets and files
        self.label_datasets_and_files = QLabel("")
        self.label_datasets_and_files.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        layout.addWidget(self.label_datasets_and_files)

    #number of channels to use
        layout_channel_selection = QHBoxLayout()
        spinbox_channel_selection = QSpinBox()
        spinbox_channel_selection .setMinimum(1)
        spinbox_channel_selection .setMaximum(7)
        spinbox_channel_selection.setValue(self.number_channels)
        select_number_channel_label = QLabel("# Channels")
        spinbox_channel_selection.valueChanged.connect(self.number_channel_changed)
        layout_channel_selection.addWidget(select_number_channel_label)
        layout_channel_selection.addWidget(spinbox_channel_selection )

        # select tracking channel

        self.spinbox_tracking_channel_selection = QSpinBox()
        self.spinbox_tracking_channel_selection.setMinimum(1)
        self.spinbox_tracking_channel_selection.setMaximum(self.number_channels)
        self.spinbox_tracking_channel_selection.setValue(self.number_channels)
        label_tracking_channel_selection = QLabel("Tracking Channel")
        # spinbox_channel_selection.valueChanged.connect(self.number_channel_changed)
        layout_channel_selection.addWidget(label_tracking_channel_selection)
        layout_channel_selection.addWidget(self.spinbox_tracking_channel_selection)
        layout.addLayout(layout_channel_selection)
    # channel names
        layout_channel_names = QVBoxLayout()
        self.channel_names = {
            1 : QLineEdit("CH1"),
            2 : QLineEdit("CH2"),
            3 : QLineEdit("CH3"),
            4 : QLineEdit("CH4"),
            5 : QLineEdit("CH5"),
            6 : QLineEdit("CH6"),
            7 : QLineEdit("CH7"),
            }
        for i, line_edit in self.channel_names.items():
            channel_layout = QHBoxLayout()
            channel_label = QLabel(F"Channel {i}")
            channel_layout.addWidget(channel_label)
            channel_layout.addWidget(line_edit)
            layout_channel_names.addLayout(channel_layout)


        layout.addLayout(layout_channel_names)

    #minimum tracking length

        layout_min_len = QHBoxLayout()
        self.spinbox_min_len = QSpinBox()
        self.spinbox_min_len.setMinimum(1)
        self.spinbox_min_len.setMaximum(999)
        self.spinbox_min_len.setValue(24)
        min_len_label = QLabel("Minimum timepoints per track")

        layout_min_len.addWidget(min_len_label)
        layout_min_len.addWidget(self.spinbox_min_len)

        self.interval_label = QLabel("Tracking interval (min)")
        self.input_interval = QLineEdit("60.0")
        layout_min_len.addWidget(self.interval_label)
        layout_min_len.addWidget(self.input_interval)

        layout.addLayout(layout_min_len)

    # advanced settings button

        self.advanced_settings_button = QPushButton('Advanced settings', self)
        self.advanced_settings_button.clicked.connect(self.open_advanced_settings)
        layout.addWidget(self.advanced_settings_button)

    # execute & Stop button
        button_layout = QHBoxLayout()
        execute_button = QPushButton("Start")
        execute_button.clicked.connect(self.execute)
        button_layout.addWidget(execute_button)

        stop_button = QPushButton("Stop")
        stop_button.clicked.connect(self.stop_execution)
        button_layout.addWidget(stop_button)
        layout.addLayout(button_layout)

    #progress_bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        layout.addWidget(self.progress_bar)

    # place layout into main window
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def open_help_window(self):
        try:
            self.help_window = HelpWindow()
            self.help_window.show()
        except:
            traceback.print_exc()

    def open_advanced_settings(self):
        try:
            self.advanced_settings_window = AdvancedSettingsWindow(self)
            self.advanced_settings_window.show()
        except:
            traceback.print_exc()

    def change_advanced_settings(self, settings_to_update):
        self.advanced_settings.update(settings_to_update)

    def stop_execution(self):
        self.keep_running = False


    def choose_folder_button_clicked(self,s):
        new_folder = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if not new_folder:
            return
        else:
            self.selected_folder = new_folder
            folder_label = ""
            for i in range(0,181,50):
                try:
                    folder_label += self.selected_folder[i:i+50]+"\n"
                except:
                    folder_label += self.selected_folder[i:]
                    break
            self.selected_folder_label.setText(F'Main Folder: {folder_label}')

            self.search_input_folder()

    def search_input_folder(self):

        def get_datasets_from_file_list(file_list, separator, digits, suffix):
            if not digits:
                #each file is an individual dataset
                return file_list
            else:
                #look for datasets (groups of files with same name pattern)
                file_list = [file.split(suffix)[0] for file in file_list]

                #check for number of digits
                try:
                    for file in file_list:
                        int(file[-int(digits):])
                except ValueError:
                    print(f'problem with file {file}')
                    raise ValueError(f'problem with file {file}\n, wrong amount of digits ?')
                file_list = [file[:-int(digits)] for file in file_list]

                #check for separator
                if separator:
                    for file in file_list:
                        if not file.endswith(separator):
                            print(f'problem with file {file}')
                            raise ValueError(f'problem with file {file}\n separator not found')
                    file_list = [separator.join(file.split(separator)[:-1]) for file in file_list]
                return list(set(file_list))

        #list search subfolders
        datasets = []
        files = [file for file in os.listdir(self.selected_folder) if os.path.isfile(self.selected_folder +"/" + file)]
        files = [file for file in files if file.endswith(self.advanced_settings["suffix"])]
        if files:
            try:
                datasets = get_datasets_from_file_list(file_list=files,
                                                       separator=self.subset_separator,
                                                       digits=self.subset_digits,
                                                       suffix = self.advanced_settings["suffix"])
                new_dataset_text = F'found {len(files)} files in {len(datasets)} datasets'
            except Exception as exp:
                print("something went wrong with file selection: ", exp)
                new_dataset_text = F"no datasets found in folder, \n{exp}"


        else:
            new_dataset_text ="no datasets found in folder,\n check Input Folder and suffix (=> Advanced settings)"
        self.label_datasets_and_files.setText(new_dataset_text)

        self.file_list = files
        self.dataset_list = datasets

    def number_channel_changed(self,new_value):
        self.number_channels=new_value
        self.spinbox_tracking_channel_selection.setMaximum(self.number_channels)

    def change_seperator(self,new_sep):
        self.subset_separator = new_sep
        self.replica_format_label.setText(self.get_replica_format())
        self.search_input_folder()

    def change_digits(self, new_number_digits):
        self.subset_digits = new_number_digits
        self.replica_format_label.setText(self.get_replica_format())
        self.search_input_folder()

    def get_replica_format(self):
        return F'{self.subset_separator}{"0"*self.subset_digits}'

    def check_inputs(self):
        if not self.dataset_list:
            raise FileNotFoundError("Found no Dataset to Process")
        try:
            float(self.input_interval.text())
        except:
            raise ValueError("Please provide valid Tracking Interval")

    def execute(self):
        self.keep_running = True
        self.errors = 0
        self.error_report = ""

        try:
            self.check_inputs()
        except Exception as e:
            self.show_error(str(e))
            return

        self.progress_bar.setValue(1)
        self.progress_window = ProgressWindow()
        self.progress_window.scroll_label.set_text("Started Processing...")
        self.progress_window.show()
        worker = Worker(self.run_main)  # Any other args, kwargs are passed to the run function
        print("worker started")
        #worker.signals.result.connect(self.print_output)
        #worker.signals.finished.connect(self.thread_complete)
        worker.signals.progress.connect(self.progress_fn)
        worker.signals.result.connect(self.result_fn)

        # Execute
        self.threadpool.start(worker)

    def progress_fn(self, n):
        #print("progress" + str(n))
        self.progress_window.progress_bar.setValue(int(100*n/len(self.dataset_list)))
        self.progress_bar.setValue(int(100*n/len(self.dataset_list)))

    def result_fn(self, results):
        result_text = F'\nprocessed {results["dataset"]}\n'
        if not results["run_complete"]:
            result_text += F'  ERROR WHILE ANALYZING DATASET\n'
            self.errors += 1
            if results["error"]:
                result_text += F'{results["error"]}\n'
                self.error_report += F'\n{results["error"]}\n'
            else:
                result_text += "UNKNOWN ERROR\n"
        else:
            result_text+= F'  approved_cells: {results["approved_cells"]}/{results["all_cells"]}\n'

        self.progress_window.scroll_label.add_text(result_text)

        if results["run_complete"]:
            self.progress_window.set_fig(results["fig_path"])

    def write_settings_file(self, output_folder):
        with open(os.path.join(output_folder, "used_settings.txt"), "w") as used_settings_file:
            settings_text = (
                f'input folder: {self.selected_folder}\n'
                f'number of channels: {self.number_channels}\n'
                f'tracking channel: {self.spinbox_tracking_channel_selection.value()}\n')
            for n in range(1,self.number_channels+1):
                settings_text += f"Name Channel {n}: {self.channel_names[n].text()}\n"
            settings_text += (
                f'minimum time points: {self.spinbox_min_len.value()}\n'
                f'tracking interval: {self.input_interval.text}\n'
                f'number of digits: {self.subset_digits}\n'
                f'delimimter: {self.subset_separator}\n')
            for setting_name, value in self.advanced_settings.items():
                settings_text += f"{setting_name}: {value}\n"
            settings_text += "\nDATASETS:\n"
            for dataset in self.dataset_list:
                settings_text+=f'  {dataset}\n'

            settings_text += "\nFILES:\n"
            for file in self.file_list:
                settings_text += f'  {file}\n'

            used_settings_file.writelines(settings_text)

    def run_main(self, progress_callback, result_callback):
        print("main started")
        min_len = self.spinbox_min_len.value()
        tracking_interval = float(self.input_interval.text())
        number_channels = self.number_channels
        channel_names = {number: self.channel_names[number].text() for number in list(range(1,number_channels+1))}
        tracking_channel = self.spinbox_tracking_channel_selection.value()
        datasets = self.dataset_list
        files = [self.selected_folder + "/" + file for file in self.file_list]
        digits = self.subset_digits
        separator = self.subset_separator
        now = datetime.datetime.now()
        y, mon, d,h,m= now.year, now.month, now.day, now.hour, now.minute
        main_output_folder = create_folder(self.selected_folder + "/" + f"grouped_results(post_script_output_"
                                                                        f"{y}-{mon:02d}-{d:02d}_{h:02d}-{m:02d})/")
        self.write_settings_file(main_output_folder)



        for i, dataset in enumerate(datasets[:]):
            # analyze dataset
            if not self.keep_running:

                time.sleep(3)
                self.progress_window.scroll_label.add_text("\nProcessing Stopped!".upper())

                return
            results = analyze_dataset(input_folder=self.selected_folder,
                            dataset_name=dataset,
                            files=files,
                            channel_dict=channel_names,
                            tracking_channel=tracking_channel,
                            min_len=min_len,
                            digits=digits,
                            delimiter = separator,
                            main_output_folder=main_output_folder,
                            advanced_settings = self.advanced_settings,
                            tracking_interval=tracking_interval)

            progress_callback.emit(int(i+1))
            result_callback.emit(results)
        time.sleep(2)
        self.progress_window.scroll_label.add_text(F"\nProcessing Finished\n {self.errors} errors occurred!")
        if self.error_report:
            self.progress_window.scroll_label.add_text(F"Errors:\n {self.error_report}")


    def show_error(self, text):
        msg = QMessageBox(parent=self)
        msg.setIcon(QMessageBox.Critical)
        msg.setText("Error")
        msg.setInformativeText(text)
        msg.setWindowTitle("Error")
        msg.exec_()
app = QApplication(sys.argv)
try:
    window = MainWindow()
    window.show()
    app.exec()
except Exception as err:
    traceback.print_exc()
    raise Exception from err