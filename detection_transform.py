#!/usr/bin/env python

import sys
import math
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge, CvBridgeError

self.image_sub = rospy.Subscriber("/cf1/camera/image_raw", Image, self.image_raw_callback)
self.info_sub = rospy.Subscriber("/cf1/camera/camera_info", CameraInfo, self.camera_info_callback)
