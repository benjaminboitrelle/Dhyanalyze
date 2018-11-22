# Dhyanalize

Script for analysing and characterising pictures taken with a CMOS camera

## Dependencies

The script has been written in Python 3.7.

System dependencies:
  * Python (3.7)
  * pip - Python package manager
  * Numpy - library for scientific computing with Python   
  * Matplotlib - 2D plotting library for Python
  * python-docx - Python library for creating and updating Microsoft Word (.docx) files (v0.8.7)

To check if you have one of these packages:
```
% pip list | grep <package>
```

To install one of the packages:
```
% pip install numpy
% pip install matplotlib
% pip install python-docx

```

N.B.: Testing procedures are underdevelopment.

## Installation

Starting from scratch:
```
% git clone https://github.com/benjaminboitrelle/Dhyanalyze.git
``` 

To update the framework:
```
% cd /path/to/dhyanalise-script
% git pull
```

## Execution

### Before starting the analysis

The script is written in such a way, that the characterisation of the dark measurements and the PTC measurements is done in one way.
The path of the data has to be like:

```
Gain mode
|  -> Dark
|    |  -> n picutres in .tif format
|    |  -> metadata file (metadata.txt)
|  -> PTC
|    |  -> Dark
|    |    |  -> n pictures in .tif format
|    |    |  -> metadata file (metadata.txt)
|    |  -> Light
|    |    |  -> n pictures in .tif format
|    |    |  -> metadata file (metadata.txt)

```
The ```metadata.txt``` contains two colmuns: [exposure time, image name]

### Run

```
% cd /path/to/dhyanalise-script
% python3 analyse_dhyana.py --help

usage: analyse_dhyana.py [-h] [--input INPUT_DIR_PATH]
                         [--output OUTPUT_DIR_PATH] [--type MEASUREMENT_TYPE]
                         [--roi [ROI]]

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT_DIR_PATH
                        Path to the input directory
  --output OUTPUT_DIR_PATH
                        Path to the output directory
  --type MEASUREMENT_TYPE
                        Type of measurements: Dark PTC
  --roi [ROI]           Select a region of interest. This ROI is squared and
                        centered. By default: 10 pixels are excluded in
                        vertically and horizontally.
 
```
