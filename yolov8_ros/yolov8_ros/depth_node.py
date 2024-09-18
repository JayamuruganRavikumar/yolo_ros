import cv2
import numpy as np
from typing import List, Tuple
import signal

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
from tf2_ros import TransformException, TransformBroadcaster
from tf2_ros.transform_listener import TransformListener

from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import TransformStamped
from yolov8_msgs.msg import Detection
from yolov8_msgs.msg import DetectionArray
from yolov8_msgs.msg import BoundingBox3D

class DepthNode(LifecycleNode):

    def __init__(self):
        super().__init__('depth_node')

        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("maximum_detection_threshold", 0.3)
        self.declare_parameter("depth_image_units_divisor", 1000)
        self.declare_parameter("bbox_inflate_factor", 0.1)
        self.declare_parameter("depth_image_reliability",
                               QoSReliabilityPolicy.BEST_EFFORT)
        self.declare_parameter("depth_info_reliability",
                               QoSReliabilityPolicy.BEST_EFFORT)
        self.tf_buffer = Buffer()
        self.cv_bridge = CvBridge()
        self.getlogget().info("Depth Node created")

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
        dinfo_reliability = self.get_parameter(
            "depth_info_reliability").get_parameter_value().integer_value

        self.depth_info_qos_profile = QoSProfile(
            reliability=dinfo_reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )
        self.bbox_inflate_factor = self.get_parameter(
            "bbox_inflate_factor").get_parameter_value().double_value
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # pubs
        self._pub = self.create_publisher(DetectionArray, "detections3D", 10)

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"Activating {self.get_name()}")

        self.depth_sub = message_filters.Subscriber(
            self, Image, "/yolo/depth_to_rgb/image_raw",
            qos_profile=self.depth_image_qos_profile)
        self.get_logger().info("Subscribed to depth_to_rgb topic")

        self.depth_info_sub = message_filters.Subscriber(
            self, CameraInfo, "/yolo/depth/camera_info",
            qos_profile=self.depth_info_qos_profile)
        self.get_logger().info("Subscribed to camera_info topic")

        self.detections_sub = message_filters.Subscriber(
            self, DetectionArray, "/yolo/detections",)
        self.get_logger().info("Subscribed to detections topic")

        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            (self.depth_sub, self.depth_info_sub, self.detections_sub), 10, 0.5)
        self._synchronizer.registerCallback(self.on_detections)

        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"Deactivating {self.get_name()}")

        #self._synchronizer.clear()
        self.depth_sub.destroy()
        self.depth_info_sub.destroy()
        self.detections_sub.destroy()

        return TransitionCallbackReturn.SUCCESS

    def on_unconfigured(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"Unconfiguring {self.get_name()}")

        return TransitionCallbackReturn.SUCCESS


    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"Cleaning up {self.get_name()}")

        del self.tf_listener
        self.destroy_publisher(self._pub)

        return TransitionCallbackReturn.SUCCESS

    def on_detections(self, depth_msg: Image, depth_info_msg: CameraInfo, detections_msg: DetectionArray):

        new_detections_msg = DetectionArray()
        new_detections_msg.header = detections_msg.header
        new_detections_msg.detections = self.process_detections(
            depth_msg, depth_info_msg, detections_msg)
        self._pub.publish(new_detections_msg)

    def process_detections(self, depth_msg: Image, depth_info_msg: CameraInfo, detections_msg: DetectionArray) -> List[Detection]:

        #Check if there are detections
        if not detections_msg.detections:
            return []
        new_detections = []
        depth_image = self.cv_bridge.imgmsg_to_cv2(depth_msg, "16UC1")

        for detection in detections_msg.detections:
            bbox3d = self.image_to_world(depth_image, depth_info_msg, detection)
            
            if bbox3d is not None:
                new_detections.append(detection)
                new_detections[-1].bbox3d = bbox3d
                self.publish_object_transform(detection, bbox3d)

        return new_detections

    def publish_object_transform(self, detection: Detection, bbox3d: BoundingBox3D):
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self.target_frame
        transform.child_frame_id = f"{detection.class_id}_{detection.id}"

        transform.transform.translation.x = bbox3d.center.position.x
        transform.transform.translation.y = bbox3d.center.position.y
        transform.transform.translation.z = bbox3d.center.position.z

        # Set rotation to identity quaternion (no rotation)
        transform.transform.rotation.x = 0.0
        transform.transform.rotation.y = 0.0
        transform.transform.rotation.z = 0.0
        transform.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(transform)

    def image_to_world(self, depth_image: np.ndarray, depth_info: CameraInfo, detection: Detection) -> BoundingBox3D:

        center_x = int(detection.bbox.center.position.x)
        center_y = int(detection.bbox.center.position.y)
        bb_width = int(detection.bbox.size.x)
        bb_height = int(detection.bbox.size.y)

        z = depth_image[center_y, center_x] 

        #Get camera intrinsic parameters
        # Camera matrix K = [fx 0 cx]
        #                   [0 fy cy]
        #                   [0 0  1 ]

        # project from image to world space
        k = depth_info.k
        cx, cy, fx, fy = k[2], k[5], k[0], k[4]
        x = z * (center_x - cx) / fx
        y = z * (center_y - cy) / fy

        # Calculate the bounding box in 3D
        world_width = z * bb_width / fx
        world_height = z * bb_height / fy
        world_depth = (world_width + world_height) / 2
        
        #Inflate the bounding box in 3D
        inflation_factor = 1.0 + self.bbox_inflate_factor
        world_width *= inflation_factor
        world_height *= inflation_factor
        world_depth *= inflation_factor

        # Update detections with depth information
        msg = BoundingBox3D()
        msg.center.position.x = x
        msg.center.position.y = y
        msg.center.position.z = float(z)
        msg.size.x = float(world_width)
        msg.size.y = float(world_height)
        msg.size.z = float(world_depth)

        return msg

def main():
    rclpy.init()
    node = DepthNode()

    node.trigger_configure()
    node.trigger_activate()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


