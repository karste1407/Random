#!/usr/bin/python

import subprocess

subprocess.check_output(['./darknet','detect','cfg/yolov3.cfg','yolov3.weights','data/horses.jpg'])
