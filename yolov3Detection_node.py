#!/usr/bin/env python

import sys
import math
import rospy
import rospkg
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import argparse

CONF_THRESH, NMS_THRESH = 0.5, 0.5

PATH_TO_NAMES = ""

net = None

output_layers = []


class detection:

    def __init__(self):
        self.detected_pub = rospy.Publisher(
            '/cf1/camera/detected_objects', Image, queue_size=2)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            "/cf1/camera/image_corrected", Image, self.callback)

    def callback(self, data):

        if data.header.stamp < (rospy.Time.now() - rospy.Duration(0.1)):
            # only take a new pictures (otherwise it will take all the old stored images in the buffer)
            # print("\nThrow away\n")
            return

        # Convert the image from rosmsg to opencv format
        try:
            cv_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        height, width = cv_img.shape[:2]

        blob = cv2.dnn.blobFromImage(
            cv_img, 0.00392, (416, 416), swapRB=True, crop=False)

        net.setInput(blob)
        layer_outputs = net.forward(output_layers)

        class_ids, confidences, b_boxes = [], [], []
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > CONF_THRESH:
                    center_x, center_y, w, h = (
                        detection[0:4] * np.array([width, height, width, height])).astype('int')

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    b_boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(int(class_id))

        if len(cv2.dnn.NMSBoxes(b_boxes, confidences, CONF_THRESH, NMS_THRESH)) > 0:
            # Perform non maximum suppression for the bounding boxes to filter overlapping and low confident bounding boxes
            indices = cv2.dnn.NMSBoxes(
                b_boxes, confidences, CONF_THRESH, NMS_THRESH).flatten().tolist()

            # Draw the filtered bounding boxes with their class to the image
            with open(PATH_TO_NAMES, "r") as f:
                classes = [line.strip() for line in f.readlines()]
            colors = np.random.uniform(0, 255, size=(len(classes), 3))

            for index in indices:
                x, y, w, h = b_boxes[index]
                cv2.rectangle(cv_img, (x, y), (x + w, y + h), colors[index], 2)
                cv2.putText(cv_img, classes[class_ids[index]], (w + 1, y + 20),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, colors[index], 2)

        try:
            self.detected_pub.publish(
                self.bridge.cv2_to_imgmsg(cv_img, "bgr8"))
        except CvBridgeError as e:
            print(e)


# Global inits
rospy.init_node('object_detection_yolov3_coco')


def main():
    global net, output_layers, PATH_TO_NAMES

    rospy.loginfo("Object detection node started! (Yolov3, Coco dataset)")

    # setup

    rospack = rospkg.RosPack()
    rospack.list()
    pkg_path = rospack.get_path("drone_marker_detection")
    param_path = pkg_path + "/params"

    PATH_TO_CFG = param_path + "/yolov3.cfg"
    PATH_TO_WEIGHTS = param_path + "/yolov3.weights"
    PATH_TO_NAMES = param_path + "/coco.names"

    # Load the network
    net = cv2.dnn.readNetFromDarknet(PATH_TO_CFG, PATH_TO_WEIGHTS)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Get the output layer from YOLO
    layers = net.getLayerNames()
    output_layers = [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    detection()  # Start image detection in openCV

    print("Object detection is running.")
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down.")

    cv2.destroyAllWindows()

    rospy.loginfo("Object detection node started was terminated.")


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
