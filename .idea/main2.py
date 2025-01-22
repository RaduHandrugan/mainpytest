import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np

class TurtleBotSpinAndMove(Node):
    def __init__(self):
        super().__init__('turtlebot_spin_and_move')

        # Publisher for velocity commands
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscribers
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)

        # Set up the image bridge for converting ROS image messages to OpenCV
        self.bridge = CvBridge()

        # Variables for sphere detection
        self.sphere_detected = False
        self.sphere_x = 0
        self.sphere_y = 0

        # Timer for spinning
        self.timer = self.create_timer(0.1, self.spin_turtlebot)  # Spin every 0.1 seconds

    def image_callback(self, msg):
        self.get_logger().info("Received an image!")  # Log when an image is received
        try:
            """Callback function to process image data from camera"""
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Convert image to HSV and filter for green color
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_green = np.array([40, 50, 50])  # Lower bound of green color
            upper_green = np.array([80, 255, 255])  # Upper bound of green color
            mask = cv2.inRange(hsv_frame, lower_green, upper_green)

            # Find contours of the green object (the sphere)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Get the largest contour which should be the sphere
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                cx = x + w // 2  # Center X of the bounding box
                cy = y + h // 2  # Center Y of the bounding box

                # Update sphere position
                self.sphere_x = cx
                self.sphere_y = cy
                self.sphere_detected = True

                # Publish the sphere's position as PoseStamped (Optional)
                pose = PoseStamped()
                pose.header.frame_id = 'camera_link'
                pose.header.stamp = self.get_clock().now().to_msg()
                pose.pose.position.x = float(self.sphere_x)  # Convert to float
                pose.pose.position.y = float(self.sphere_y)  # Convert to float
                pose.pose.position.z = 0.0  # Already a float
                self.get_logger().info(f"Green sphere detected at: x={cx}, y={cy}")

                # Move towards the sphere
                self.move_towards_sphere()

        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")

    def move_towards_sphere(self):
        """Move the robot towards the detected green sphere."""
        msg = Twist()

        # Control robot's movement using simple proportional control (adjust scaling as needed)
        if self.sphere_x < 300:  # Detect if the robot needs to adjust left
            msg.angular.z = 0.1  # Turn left
        elif self.sphere_x > 340:  # Detect if the robot needs to adjust right
            msg.angular.z = -0.1  # Turn right
        else:
            msg.linear.x = 0.2  # Move forward

        self.publisher.publish(msg)

        # Optionally stop if the robot is near the sphere (you can tune this condition)
        if self.sphere_x > 320 and self.sphere_x < 330:
            msg.linear.x = 0.0  # Stop moving forward
            msg.angular.z = 0.0  # Stop rotating
            self.publisher.publish(msg)
            self.get_logger().info("Approached the green sphere. Stopping robot.")

    def spin_turtlebot(self):
        """Spin the bot around the green sphere while keeping focus."""
        if not self.sphere_detected:
            self.search_for_sphere()
            return

        # If the sphere is detected, stop spinning and move towards it
        self.get_logger().info("Sphere detected, moving towards it.")
        self.move_towards_sphere()

    def search_for_sphere(self):
        """Rotate to search for the green sphere."""
        self.get_logger().info("Searching for the green sphere...")
        msg = Twist()
        msg.angular.z = 1.0  # Rotate to search
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TurtleBotSpinAndMove()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()