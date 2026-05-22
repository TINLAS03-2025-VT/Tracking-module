# Tracking Module

## Intro
In this repo the apriltags based localization code for camera based tracking of the robots resides.

## Commands
For the commands, enter a python venv. Install the requirements given in the requirements.txt. Then change directory to the main project dir.

- Start 30 seconds calibration for a 10x7 checkerboard pattern size 23mm at camera device 5: `python src/calibrate_camera.py -r 6 -c 9 -s 23 -i 5`


- Start the localization: `python src/main`