# GUI.Analysing-Cell-Tracking-Data
 A GUI to analyze data from cell Tracking using ImageJ/Trackmate


This Python based GUI is designed to load in (high troughput) tracking data generated with Fiji/TrackMate.
The data is then quality controlled (see below) and filtered according to the inputs.
Outputs as an excel file with the per-cell data, a visual overview, and a settings-used file.

Dependencies:
_____________
	
- Python Version test: 3.9.2

- Packages to be installed:
		- Numpy
		- Pandas
		- PyQT5
		- Openpyxl
		- Matplotlib

To test the GUI, start the GUI, download the sample data, choose the download folder as input folder, and press start.
Furter details are givin in the help window. 

Sample Data:
____________

Cells were imaged on a widefield microscope for 3 days with one pic/h in multiple colors. 
Channel 1 detects CRY1-Fusion protein, whose expression (intensity) oscillated over the course of a day
Channel 2 is empty
Channel 3 detects a nucelar tracking marker (irfp720-H2B)
Channel 4 is phase contrast
See sample videos.

The raw data was tracked in FIJI/ImageJ using TrackMate with StarDist segmentation and LAP tracking.
Sample data contains traking results from 8 different fields of view from two different settings as csv files.


Principle of quality control
____________________________

Errors that usually occur within cell tracking are wrong segmentation (e.g. two cells appear as one cell) and wrong tracking 
(connecting the wrong cells between time points). Both can often be detected by apprupt changes in cell/nucleus (=object) size. 
However, those changes in object size also occur during cell division.
Thus, the script first detects cell divisions as characteristic changes in cell size (first smaller, then larger) accompanied by 
a characteristic peak in the tracking marker intensity that also occurs during division. Than, object size jumps not related to cell
division are detected as errors and erroneous tracks are sorted out.
Parameters for this detection can be adjusted in "advanced settings"
For more details, refer to https://www.pnas.org/doi/10.1073/pnas.2404738121 
