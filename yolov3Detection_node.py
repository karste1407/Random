
import sys
import math
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import argparse

PATH_TO_CFG = 'darknet/cfg/yolov3.cfg'
PATH_TO_WEIGHTS = 'darknet/yolov3.weights'
PATH_TO_NAMES = 'darknet/data/coco.names'

CONF_THRESH, NMS_THRESH = 0.5, 0.5

parser = argparse.ArgumentParser(add_help=False)
#parser.add_argument("--image", default='custom_data/test_images/stop_sign_15.jpg', help="image for prediction")
parser.add_argument("--config", default=PATH_TO_CFG, help="YOLO config path")
parser.add_argument("--weights", default=PATH_TO_WEIGHTS, help="YOLO weights path")
parser.add_argument("--names", default=PATH_TO_NAMES, help="class names path")
args = parser.parse_args()

# Load the network
net = cv2.dnn.readNetFromDarknet(args.config, args.weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Get the output layer from YOLO
layers = net.getLayerNames()
output_layers = [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]


class detection:

    def __init__(self):
        self.detected_pub = rospy.Publisher('/cf1/camera/detected_objects',Image,queue_size=2)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            "/cf1/camera/image_corrected", Image, self.callback)

    def callback(self, data):
        global counter, incrementer, incrementer_updater

        # Convert the image from rosmsg to opencv format
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        height, width = cv_img.shape[:2]

        blob = cv2.dnn.blobFromImage(cv_img, 0.00392, (416, 416), swapRB=True, crop=False)

        net.setInput(blob)
        layer_outputs = net.forward(output_layers)

        class_ids, confidences, b_boxes = [], [], []
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > CONF_THRESH:
                    center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    b_boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(int(class_id))

        # Perform non maximum suppression for the bounding boxes to filter overlapping and low confident bounding boxes
        #print(cv2.dnn.NMSBoxes(b_boxes, confidences, CONF_THRESH, NMS_THRESH))
        indices = cv2.dnn.NMSBoxes(b_boxes, confidences, CONF_THRESH, NMS_THRESH).flatten().tolist()

        # Draw the filtered bounding boxes with their class to the image
        with open(args.names, "r") as f:
            classes = [line.strip() for line in f.readlines()]
        colors = np.random.uniform(0, 255, size=(len(classes), 3))

        for index in indices:
            x, y, w, h = b_boxes[index]
            cv2.rectangle(cv_img, (x, y), (x + w, y + h), colors[index], 2)
            cv2.putText(cv_img, classes[class_ids[index]], (x + 5, y + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, colors[index], 2)

        try:
            self.detected_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        except CvBridgeError as e:
            print(e)


# Global inits
rospy.init_node('object_detection_yolov3_coco')


def main():

    rospy.loginfo("Object detection node started! (Yolov3, Coco dataset ")

    ic = detection()  # Start image detection in openCV

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
