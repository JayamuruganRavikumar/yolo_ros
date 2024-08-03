import cv2
import numpy as np
from typing import List, Tuple

import rclpy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState

import message_filters
from cv_bridge import CvBridge
from tf2_ros.buffer import Buffer
from tf2_ros import TransformException
from tf2_ros.transform_listener import TransformListener

from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import TransformStamped
from yolov8_msgs.msg import Detection
from yolov8_msgs.msg import DetectionArray

class DepthNode(LifecycleNode):

    def __init__(self):
        super().__init__('depth_node')

        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("maximum_detection_threshold", 0.3)
        self.declare_parameter("depth_image_units_divisor", 1000)
        self.declare_parameter("depth_image_reliability",
                               QoSReliabilityPolicy.BEST_EFFORT)
        self.declare_parameter("depth_info_reliability",
                               QoSReliabilityPolicy.BEST_EFFORT)
        self.tf_buffer = Buffer()
        self.cv_bridge = CvBridge()

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        
        self.get_logger().info(f"Configuring {self.get_name()}")

        self.target_frame = self.get_parameter(
            "target_frame").get_parameter_value().string_value
        self.maximum_detection_threshold = self.get_parameter(
            "maximum_detection_threshold").get_parameter_value().double_value
        self.depth_image_units_divisor = self.get_parameter(
            "depth_image_units_divisor").get_parameter_value().integer_value
        dimg_reliability = self.get_parameter(
            "depth_image_reliability").get_parameter_value().integer_value

        self.depth_image_qos_profile = QoSProfile(
            reliability=dimg_reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # pubs
        self._pub = self.create_publisher(DetectionArray, "detections_position", 10)

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"Activating {self.get_name()}")

        self.depth_sub = message_filters.Subscriber(
            self, Image, "depth_image",
            qos_profile=self.depth_image_qos_profile)
        self.depth_info_sub = message_filters.Subscriber(
            self, CameraInfo, "depth_info",
            qos_profile=self.depth_info_qos_profile)
        self.detections_sub = message_filters.Subscriber(
            self, DetectionArray, "detections")

        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            (self.depth_sub, self.depth_info_sub, self.detections_sub), 10, 0.5)
        self._synchronizer.registerCallback(self.on_detections)

        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"Deactivating {self.get_name()}")

        self._synchronizer.clear()
        self.depth_sub.destroy()
        self.depth_info_sub.destroy()
        self.detections_sub.destroy()

        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"Cleaning up {self.get_name()}")

        del self.tf_listener
        self.destroy_publisher(self._pub)

        return TransitionCallbackReturn.SUCCESS

    def on_detections(self, depth_image: Image, depth_info: CameraInfo, detections: DetectionArray):


