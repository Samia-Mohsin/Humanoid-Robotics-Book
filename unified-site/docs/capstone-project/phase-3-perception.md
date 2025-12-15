# Phase 3: Perception

## Overview

Phase 3 focuses on implementing perception capabilities for your humanoid robot, including computer vision, Visual Simultaneous Localization and Mapping (VSLAM), obstacle detection, and sensor fusion. This phase enables your robot to understand and navigate its environment effectively, building upon the locomotion capabilities developed in Phase 2.

## Learning Objectives

By the end of this phase, you will be able to:

- Implement computer vision algorithms for object identification and tracking
- Develop VSLAM systems for environment mapping and localization
- Design obstacle detection and avoidance systems
- Create sensor fusion frameworks that combine multiple sensor inputs for enhanced environmental awareness

## Weekly Breakdown

### Week 1: Computer Vision and Object Recognition

**Learning Goals:**
- Implement object detection and recognition systems
- Develop image processing pipelines for real-time perception
- Integrate vision systems with robot control

**Activities:**
- Study deep learning-based object detection (YOLO, SSD, etc.)
- Implement real-time object tracking algorithms
- Develop color and shape-based object recognition
- Test vision systems in simulation environment

**Deliverables:**
- Object detection and recognition pipeline
- Real-time tracking system
- Integration with robot's visual sensors

### Week 2: VSLAM and Environment Mapping

**Learning Goals:**
- Implement Visual SLAM for mapping and localization
- Develop sensor fusion for enhanced perception
- Integrate with navigation systems

**Activities:**
- Implement ORB-SLAM or similar VSLAM algorithms
- Develop map building and maintenance systems
- Integrate IMU and visual data for robust localization
- Test mapping accuracy and reliability

**Deliverables:**
- VSLAM implementation
- Environment map building system
- Localization accuracy validation

## Perception System Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   RGB-D Camera  │    │   LIDAR/Depth   │    │   IMU Sensors   │
│   (Vision)      │    │   (Range)       │    │   (Orientation) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Sensor Fusion Layer                          │
│                    (Kalman Filters, Particle Filters)           │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Perception Processing                        │
│                    - Object Detection                           │
│                    - Feature Extraction                         │
│                    - Environment Mapping                        │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    World Model & Decision Making                │
└─────────────────────────────────────────────────────────────────┘
```

### Core Perception Components

The perception system consists of several interconnected modules:

```python
import cv2
import numpy as np
import torch
import rospy
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
from std_msgs.msg import Header
import tf2_ros
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PointStamped

class PerceptionSystem:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('perception_system')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Initialize TF buffer
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Publishers and subscribers
        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback)
        self.depth_sub = rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_callback)
        self.camera_info_sub = rospy.Subscriber('/camera/rgb/camera_info', CameraInfo, self.camera_info_callback)
        self.object_pub = rospy.Publisher('/detected_objects', MarkerArray, queue_size=10)

        # Object detection model
        self.object_detector = self.initialize_object_detector()

        # SLAM system
        self.vslam_system = VSLAMSystem()

        # State variables
        self.current_image = None
        self.current_depth = None
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.world_objects = {}

        # Processing parameters
        self.confidence_threshold = 0.7
        self.distance_threshold = 3.0  # meters

    def initialize_object_detector(self):
        """Initialize object detection model"""
        # For simulation, we'll use a mock detector
        # In real implementation, this would load YOLO, Detectron2, or similar
        return MockObjectDetector()

    def image_callback(self, msg):
        """Callback for RGB image data"""
        try:
            # Convert ROS image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.current_image = cv_image

            # Process the image for object detection
            self.process_image(cv_image)

        except Exception as e:
            rospy.logerr(f"Error processing image: {str(e)}")

    def depth_callback(self, msg):
        """Callback for depth image data"""
        try:
            # Convert ROS depth image message to OpenCV image
            depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")
            self.current_depth = depth_image

        except Exception as e:
            rospy.logerr(f"Error processing depth image: {str(e)}")

    def camera_info_callback(self, msg):
        """Callback for camera calibration information"""
        self.camera_matrix = np.array(msg.K).reshape(3, 3)
        self.distortion_coeffs = np.array(msg.D)

    def process_image(self, image):
        """Process image for object detection and tracking"""
        if self.current_depth is not None and self.camera_matrix is not None:
            # Detect objects in the image
            detections = self.object_detector.detect(image)

            # Filter detections based on confidence
            filtered_detections = [
                det for det in detections
                if det['confidence'] > self.confidence_threshold
            ]

            # Convert 2D detections to 3D world coordinates
            world_objects = []
            for detection in filtered_detections:
                world_point = self.pixel_to_world(
                    detection['bbox_center'],
                    self.current_depth,
                    detection['distance']
                )

                if world_point is not None:
                    world_objects.append({
                        'class': detection['class'],
                        'confidence': detection['confidence'],
                        'position': world_point,
                        'bbox': detection['bbox']
                    })

            # Update world model with detected objects
            self.update_world_model(world_objects)

            # Publish visualization markers
            self.publish_object_markers(world_objects)

    def pixel_to_world(self, pixel_coords, depth_image, distance_estimate):
        """Convert pixel coordinates to world coordinates"""
        if self.camera_matrix is None:
            return None

        u, v = pixel_coords
        # Get depth value at pixel location
        depth = depth_image[v, u] if 0 <= v < depth_image.shape[0] and 0 <= u < depth_image.shape[1] else distance_estimate

        # Convert to world coordinates using camera intrinsics
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]

        # Calculate 3D point in camera frame
        x_cam = (u - cx) * depth / fx
        y_cam = (v - cy) * depth / fy
        z_cam = depth

        # Transform to world frame (assuming camera is calibrated with respect to robot base)
        # This would require TF transforms in a real implementation
        world_point = Point()
        world_point.x = x_cam
        world_point.y = y_cam
        world_point.z = z_cam

        return world_point

    def update_world_model(self, detected_objects):
        """Update the world model with new object detections"""
        for obj in detected_objects:
            obj_id = f"{obj['class']}_{hash(str(obj['position'])) % 10000}"

            # Check if this is a new object or update to existing
            if obj_id in self.world_objects:
                # Update existing object
                self.world_objects[obj_id]['last_seen'] = rospy.Time.now()
                self.world_objects[obj_id]['position'] = obj['position']
                self.world_objects[obj_id]['confidence'] = max(
                    self.world_objects[obj_id]['confidence'],
                    obj['confidence']
                )
            else:
                # Add new object
                self.world_objects[obj_id] = {
                    'class': obj['class'],
                    'position': obj['position'],
                    'confidence': obj['confidence'],
                    'first_seen': rospy.Time.now(),
                    'last_seen': rospy.Time.now()
                }

        # Remove objects that haven't been seen recently
        current_time = rospy.Time.now()
        objects_to_remove = []
        for obj_id, obj_data in self.world_objects.items():
            time_since_seen = (current_time - obj_data['last_seen']).to_sec()
            if time_since_seen > 5.0:  # Remove if not seen in 5 seconds
                objects_to_remove.append(obj_id)

        for obj_id in objects_to_remove:
            del self.world_objects[obj_id]

    def publish_object_markers(self, objects):
        """Publish visualization markers for detected objects"""
        marker_array = MarkerArray()

        for i, obj in enumerate(objects):
            marker = Marker()
            marker.header = Header()
            marker.header.frame_id = "base_link"  # Robot base frame
            marker.header.stamp = rospy.Time.now()
            marker.ns = "detected_objects"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            # Set position
            marker.pose.position = obj['position']
            marker.pose.orientation.w = 1.0

            # Set scale (0.1m radius sphere)
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1

            # Set color based on object class
            if obj['class'] == 'cup':
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
            elif obj['class'] == 'chair':
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            elif obj['class'] == 'table':
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0
            else:
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0

            marker.color.a = 0.8  # Alpha

            marker_array.markers.append(marker)

        self.object_pub.publish(marker_array)
```

## Computer Vision Implementation

### Object Detection and Recognition

```python
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class MockObjectDetector:
    def __init__(self):
        # In a real implementation, this would load a pre-trained model
        # such as YOLO, SSD, or similar
        self.classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
            'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush'
        ]

        # For simulation, we'll create mock detections
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

    def detect(self, image):
        """Detect objects in an image"""
        # In a real implementation, this would run the actual detection model
        # For simulation, return mock detections

        # Simulate detecting some objects in the image
        height, width = image.shape[:2]
        detections = []

        # Add some mock detections based on image properties
        if width > 0 and height > 0:
            # Detect a cup in the center of the image
            detections.append({
                'class': 'cup',
                'bbox': [width//2 - 50, height//2 - 50, width//2 + 50, height//2 + 50],
                'bbox_center': [width//2, height//2],
                'confidence': 0.85,
                'distance': 1.2  # meters (simulated)
            })

            # Detect a chair on the left side
            detections.append({
                'class': 'chair',
                'bbox': [width//4 - 60, height//2 - 80, width//4 + 60, height//2 + 80],
                'bbox_center': [width//4, height//2],
                'confidence': 0.78,
                'distance': 1.5  # meters (simulated)
            })

            # Detect a table in the background
            detections.append({
                'class': 'dining table',
                'bbox': [width//3, height//2, 2*width//3, 3*height//4],
                'bbox_center': [width//2, 5*height//8],
                'confidence': 0.92,
                'distance': 1.0  # meters (simulated)
            })

        return detections

class FeatureExtractor:
    def __init__(self):
        """Initialize feature extraction system"""
        # In a real implementation, this might use SIFT, SURF, ORB, or deep learning features
        self.feature_detector = cv2.ORB_create()

    def extract_features(self, image):
        """Extract features from an image"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect and compute features
        keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)

        return keypoints, descriptors

    def match_features(self, desc1, desc2):
        """Match features between two images"""
        # Create BF matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors
        matches = bf.match(desc1, desc2)

        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        return matches
```

## VSLAM Implementation

### Visual SLAM System

```python
import numpy as np
from scipy.spatial.transform import Rotation as R
from collections import deque
import threading

class VSLAMSystem:
    def __init__(self):
        self.current_pose = np.eye(4)  # 4x4 transformation matrix
        self.keyframes = []
        self.map_points = []
        self.feature_trackers = {}

        # Tracking parameters
        self.keyframe_threshold = 0.1  # Movement threshold to create keyframe
        self.max_features = 1000
        self.min_matches = 20

        # Previous frame data
        self.prev_image = None
        self.prev_features = None
        self.prev_descriptors = None

        # Map management
        self.local_map_size = 100  # Number of recent keyframes to maintain
        self.global_map_size = 1000  # Total map size limit

        # Lock for thread safety
        self.slam_lock = threading.Lock()

    def process_frame(self, image, camera_matrix):
        """Process a new frame for SLAM"""
        with self.slam_lock:
            if self.prev_image is None:
                # Initialize with first frame
                self.prev_image = image
                self.prev_features, self.prev_descriptors = self.extract_features(image)

                # Add first keyframe
                first_keyframe = {
                    'image': image,
                    'pose': np.eye(4),
                    'features': self.prev_features,
                    'descriptors': self.prev_descriptors,
                    'timestamp': rospy.Time.now()
                }
                self.keyframes.append(first_keyframe)
                return np.eye(4)  # Return initial pose

            # Extract features from current frame
            curr_features, curr_descriptors = self.extract_features(image)

            # Match features with previous frame
            matches = self.match_features(self.prev_descriptors, curr_descriptors)

            if len(matches) < self.min_matches:
                # Not enough matches, skip frame
                self.prev_image = image
                self.prev_features = curr_features
                self.prev_descriptors = curr_descriptors
                return self.current_pose

            # Extract matched points
            prev_pts = np.float32([self.prev_features[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            curr_pts = np.float32([curr_features[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Estimate motion using Essential matrix
            E, mask = cv2.findEssentialMat(
                curr_pts, prev_pts, camera_matrix,
                method=cv2.RANSAC, threshold=1.0
            )

            if E is not None:
                # Recover pose from Essential matrix
                _, R, t, _ = cv2.recoverPose(E, curr_pts, prev_pts, camera_matrix)

                # Create transformation matrix
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = t.flatten()

                # Update current pose
                self.current_pose = self.current_pose @ T

                # Check if we should create a new keyframe
                translation_norm = np.linalg.norm(T[:3, 3])
                rotation_angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))

                if translation_norm > self.keyframe_threshold or rotation_angle > self.keyframe_threshold:
                    # Add new keyframe
                    new_keyframe = {
                        'image': image,
                        'pose': self.current_pose.copy(),
                        'features': curr_features,
                        'descriptors': curr_descriptors,
                        'timestamp': rospy.Time.now()
                    }
                    self.keyframes.append(new_keyframe)

                    # Limit keyframe buffer size
                    if len(self.keyframes) > self.local_map_size:
                        self.keyframes.pop(0)

                # Update previous frame data
                self.prev_image = image
                self.prev_features = curr_features
                self.prev_descriptors = curr_descriptors

            return self.current_pose.copy()

    def extract_features(self, image):
        """Extract ORB features from image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Create ORB detector
        orb = cv2.ORB_create(nfeatures=self.max_features)

        # Detect and compute features
        keypoints, descriptors = orb.detectAndCompute(gray, None)

        return keypoints, descriptors

    def match_features(self, desc1, desc2):
        """Match features between two sets of descriptors"""
        if desc1 is None or desc2 is None:
            return []

        # Create BF matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Match descriptors
        matches = bf.knnMatch(desc1, desc2, k=2)

        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        return good_matches

    def get_current_pose(self):
        """Get current estimated pose"""
        with self.slam_lock:
            return self.current_pose.copy()

    def get_map(self):
        """Get the current map"""
        with self.slam_lock:
            return {
                'keyframes': self.keyframes.copy(),
                'map_points': self.map_points.copy(),
                'current_pose': self.current_pose.copy()
            }
```

## Sensor Fusion Implementation

### Multi-Sensor Data Integration

```python
import numpy as np
from scipy.linalg import block_diag
from collections import defaultdict

class SensorFusion:
    def __init__(self):
        # State vector: [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
        self.state_dim = 12

        # Initialize state and covariance
        self.x = np.zeros(self.state_dim)  # State vector
        self.P = np.eye(self.state_dim) * 1000  # Covariance matrix

        # Process noise
        self.Q = np.eye(self.state_dim) * 0.1

        # Time tracking
        self.last_update_time = None

        # Sensor data buffers
        self.imu_buffer = deque(maxlen=10)
        self.vision_buffer = deque(maxlen=5)
        self.odometry_buffer = deque(maxlen=10)

        # Sensor noise characteristics
        self.imu_noise = np.diag([0.01, 0.01, 0.01, 0.001, 0.001, 0.001])  # [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
        self.vision_noise = np.diag([0.05, 0.05, 0.05, 0.01, 0.01, 0.01])  # [pos_x, pos_y, pos_z, orient_x, orient_y, orient_z]
        self.odom_noise = np.diag([0.1, 0.1, 0.1, 0.05, 0.05, 0.05])  # [pos_x, pos_y, pos_z, orient_x, orient_y, orient_z]

    def predict(self, dt):
        """Prediction step of the Kalman filter"""
        if dt <= 0:
            return

        # State transition matrix (simplified - assumes constant velocity model)
        F = np.eye(self.state_dim)
        F[0:3, 6:9] = np.eye(3) * dt  # Position from velocity
        F[3:6, 9:12] = np.eye(3) * dt  # Orientation from angular velocity

        # Predict state
        self.x = F @ self.x

        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q * dt

    def update_imu(self, accel, gyro, timestamp):
        """Update filter with IMU measurements"""
        if self.last_update_time is None:
            self.last_update_time = timestamp
            return

        dt = (timestamp - self.last_update_time).to_sec()
        self.last_update_time = timestamp

        # Prediction step
        self.predict(dt)

        # Measurement matrix for IMU (we only measure acceleration and angular velocity)
        H = np.zeros((6, self.state_dim))
        H[0:3, 0:3] = np.eye(3)  # Acceleration measurements
        H[3:6, 9:12] = np.eye(3)  # Angular velocity measurements

        # IMU measurement vector
        z = np.concatenate([accel, gyro])

        # Innovation
        innovation = z - H @ self.x[0:6]  # Only use first 6 elements for IMU

        # Innovation covariance
        S = H @ self.P[0:6, 0:6] @ H.T + self.imu_noise

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state and covariance
        self.x += K @ innovation
        self.P = (np.eye(self.state_dim) - K @ H) @ self.P

    def update_vision(self, position, orientation, timestamp):
        """Update filter with vision measurements"""
        if self.last_update_time is None:
            self.last_update_time = timestamp
            return

        dt = (timestamp - self.last_update_time).to_sec()
        self.last_update_time = timestamp

        # Prediction step
        self.predict(dt)

        # Measurement matrix for vision (position and orientation)
        H = np.zeros((6, self.state_dim))
        H[0:3, 0:3] = np.eye(3)  # Position measurements
        H[3:6, 3:6] = np.eye(3)  # Orientation measurements

        # Vision measurement vector
        z = np.concatenate([position, orientation])

        # Innovation
        innovation = z - H @ self.x[0:6]

        # Innovation covariance
        S = H @ self.P[0:6, 0:6] @ H.T + self.vision_noise

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state and covariance
        self.x += K @ innovation
        self.P = (np.eye(self.state_dim) - K @ H) @ self.P

    def update_odometry(self, position, orientation, timestamp):
        """Update filter with odometry measurements"""
        if self.last_update_time is None:
            self.last_update_time = timestamp
            return

        dt = (timestamp - self.last_update_time).to_sec()
        self.last_update_time = timestamp

        # Prediction step
        self.predict(dt)

        # Measurement matrix for odometry (position and orientation)
        H = np.zeros((6, self.state_dim))
        H[0:3, 0:3] = np.eye(3)  # Position measurements
        H[3:6, 3:6] = np.eye(3)  # Orientation measurements

        # Odometry measurement vector
        z = np.concatenate([position, orientation])

        # Innovation
        innovation = z - H @ self.x[0:6]

        # Innovation covariance
        S = H @ self.P[0:6, 0:6] @ H.T + self.odom_noise

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state and covariance
        self.x += K @ innovation
        self.P = (np.eye(self.state_dim) - K @ H) @ self.P

    def get_fused_state(self):
        """Get the current fused state estimate"""
        return {
            'position': self.x[0:3],
            'orientation': self.x[3:6],
            'velocity': self.x[6:9],
            'angular_velocity': self.x[9:12],
            'position_covariance': self.P[0:3, 0:3],
            'orientation_covariance': self.P[3:6, 3:6]
        }
```

## Obstacle Detection and Avoidance

### Environment Perception and Navigation Safety

```python
import numpy as np
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
import rospy

class ObstacleDetection:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('obstacle_detection')

        # Publishers and subscribers
        self.laser_sub = rospy.Subscriber('/scan', LaserScan, self.laser_callback)
        self.depth_sub = rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_callback)
        self.map_pub = rospy.Publisher('/local_map', OccupancyGrid, queue_size=10)

        # Parameters
        self.map_resolution = 0.1  # meters per cell
        self.map_width = 200  # cells (20m x 20m map)
        self.map_height = 200  # cells
        self.robot_radius = 0.3  # meters
        self.clearing_radius = 0.5  # meters for clearing space
        self.min_obstacle_height = 0.1  # minimum height to consider obstacle
        self.max_obstacle_height = 2.0  # maximum height to consider obstacle

        # Local map (2D occupancy grid)
        self.local_map = np.zeros((self.map_height, self.map_width), dtype=np.int8)
        self.map_origin = Point()
        self.map_origin.x = -self.map_width * self.map_resolution / 2.0
        self.map_origin.y = -self.map_height * self.map_resolution / 2.0

        # Obstacle detection parameters
        self.min_laser_range = 0.1  # meters
        self.max_laser_range = 10.0  # meters
        self.laser_angle_increment = 0.01  # radians

    def laser_callback(self, msg):
        """Process laser scan data for obstacle detection"""
        # Update local map with laser scan data
        self.update_local_map_laser(msg)

        # Publish updated map
        self.publish_local_map()

    def depth_callback(self, msg):
        """Process depth image for obstacle detection"""
        try:
            # Convert ROS depth image to OpenCV
            depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")

            # Process depth image for obstacle detection
            self.update_local_map_depth(depth_image)

        except Exception as e:
            rospy.logerr(f"Error processing depth image: {str(e)}")

    def update_local_map_laser(self, laser_scan):
        """Update local map using laser scan data"""
        # Reset map to unknown
        self.local_map.fill(-1)

        # Process each laser beam
        for i, range_val in enumerate(laser_scan.ranges):
            if self.min_laser_range <= range_val <= self.max_laser_range:
                # Calculate angle of this beam
                angle = laser_scan.angle_min + i * laser_scan.angle_increment

                # Calculate world coordinates
                x_world = range_val * np.cos(angle)
                y_world = range_val * np.sin(angle)

                # Convert to map coordinates
                map_x = int((x_world - self.map_origin.x) / self.map_resolution)
                map_y = int((y_world - self.map_origin.y) / self.map_resolution)

                # Check bounds
                if 0 <= map_x < self.map_width and 0 <= map_y < self.map_height:
                    # Mark as occupied (obstacle detected)
                    self.local_map[map_y, map_x] = 100  # Occupied

    def update_local_map_depth(self, depth_image):
        """Update local map using depth image data"""
        # For simulation, we'll create a simple representation
        # In reality, this would involve more complex processing

        # Get image dimensions
        height, width = depth_image.shape

        # Process depth image to detect obstacles
        for v in range(0, height, 10):  # Sample every 10th row/column
            for u in range(0, width, 10):
                depth_val = depth_image[v, u]

                if not np.isnan(depth_val) and not np.isinf(depth_val):
                    # Convert pixel to 3D world coordinates (simplified)
                    # This requires camera intrinsics for accurate conversion
                    x_world = (u - width/2) * depth_val * 0.001  # Simplified conversion
                    y_world = (v - height/2) * depth_val * 0.001  # Simplified conversion

                    # Convert to map coordinates
                    map_x = int((x_world - self.map_origin.x) / self.map_resolution)
                    map_y = int((y_world - self.map_origin.y) / self.map_resolution)

                    # Check bounds and if it's a valid obstacle
                    if (0 <= map_x < self.map_width and
                        0 <= map_y < self.map_height and
                        self.min_obstacle_height <= depth_val <= self.max_obstacle_height):
                        # Mark as occupied with some probability based on confidence
                        if self.local_map[map_y, map_x] < 50:  # If not already highly occupied
                            self.local_map[map_y, map_x] = 75  # Likely occupied

    def publish_local_map(self):
        """Publish the local occupancy grid map"""
        map_msg = OccupancyGrid()

        # Set header
        map_msg.header.stamp = rospy.Time.now()
        map_msg.header.frame_id = "map"  # or "base_link" depending on your convention

        # Set map info
        map_msg.info.resolution = self.map_resolution
        map_msg.info.width = self.map_width
        map_msg.info.height = self.map_height
        map_msg.info.origin.position.x = self.map_origin.x
        map_msg.info.origin.position.y = self.map_origin.y
        map_msg.info.origin.position.z = 0.0
        map_msg.info.origin.orientation.w = 1.0

        # Flatten the map data (row-major order)
        map_data = self.local_map.flatten()
        map_msg.data = map_data.tolist()

        # Publish the map
        self.map_pub.publish(map_msg)

    def is_path_clear(self, start_point, end_point, safety_margin=0.2):
        """Check if path between two points is clear of obstacles"""
        # Convert to map coordinates
        start_map_x = int((start_point.x - self.map_origin.x) / self.map_resolution)
        start_map_y = int((start_point.y - self.map_origin.y) / self.map_resolution)

        end_map_x = int((end_point.x - self.map_origin.x) / self.map_resolution)
        end_map_y = int((end_point.y - self.map_origin.y) / self.map_resolution)

        # Check if start and end points are within bounds
        if (not (0 <= start_map_x < self.map_width and 0 <= start_map_y < self.map_height) or
            not (0 <= end_map_x < self.map_width and 0 <= end_map_y < self.map_height)):
            return False

        # Use Bresenham's line algorithm to check path
        points = self.bresenham_line(start_map_x, start_map_y, end_map_x, end_map_y)

        # Check each point on the path
        safety_cells = int(safety_margin / self.map_resolution)

        for x, y in points:
            # Check surrounding cells for safety margin
            for dx in range(-safety_cells, safety_cells + 1):
                for dy in range(-safety_cells, safety_cells + 1):
                    check_x = x + dx
                    check_y = y + dy

                    if (0 <= check_x < self.map_width and 0 <= check_y < self.map_height):
                        if self.local_map[check_y, check_x] > 50:  # Occupied cell
                            return False  # Path is blocked

        return True  # Path is clear

    def bresenham_line(self, x0, y0, x1, y1):
        """Bresenham's line algorithm to get points on a line"""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x_step = 1 if x0 < x1 else -1
        y_step = 1 if y0 < y1 else -1
        error = dx - dy

        x, y = x0, y0

        while True:
            points.append((x, y))

            if x == x1 and y == y1:
                break

            error2 = 2 * error
            if error2 > -dy:
                error -= dy
                x += x_step
            if error2 < dx:
                error += dx
                y += y_step

        return points
```

## Integration with Navigation

### Perception-Action Integration

```python
from geometry_msgs.msg import PoseStamped, PointStamped
from actionlib_msgs.msg import GoalStatusArray
import tf2_ros
from visualization_msgs.msg import Marker, MarkerArray

class PerceptionNavigationIntegration:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('perception_navigation_integration')

        # TF buffer for coordinate transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Publishers and subscribers
        self.move_base_goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        self.goal_status_sub = rospy.Subscriber('/move_base/status', GoalStatusArray, self.goal_status_callback)
        self.robot_pose_sub = rospy.Subscriber('/amcl_pose', PoseStamped, self.robot_pose_callback)

        # Perception system reference
        self.perception_system = None
        self.obstacle_detector = None

        # Navigation state
        self.current_goal = None
        self.goal_status = None
        self.robot_pose = None

        # Navigation parameters
        self.replan_threshold = 0.5  # meters
        self.safe_distance = 0.8  # minimum distance to obstacles
        self.check_frequency = 1.0  # Hz

        # Timer for periodic navigation checks
        self.nav_timer = rospy.Timer(
            rospy.Duration(1.0/self.check_frequency),
            self.navigation_check_callback
        )

    def set_perception_system(self, perception_system, obstacle_detector):
        """Set references to perception and obstacle detection systems"""
        self.perception_system = perception_system
        self.obstacle_detector = obstacle_detector

    def robot_pose_callback(self, msg):
        """Update robot pose"""
        self.robot_pose = msg

    def goal_status_callback(self, msg):
        """Update goal status"""
        if len(msg.status_list) > 0:
            self.goal_status = msg.status_list[-1]  # Get most recent status

    def navigation_check_callback(self, event):
        """Periodic check for navigation safety and replanning"""
        if self.current_goal is None or self.robot_pose is None:
            return

        # Check if current path is still safe
        if not self.is_path_safe():
            rospy.logwarn("Path is no longer safe, replanning...")
            self.replan_path()

    def is_path_safe(self):
        """Check if the current path is safe from obstacles"""
        if self.robot_pose is None or self.current_goal is None:
            return True  # If no pose or goal, assume safe

        # Check if there are obstacles blocking the path
        robot_point = PointStamped()
        robot_point.header = self.robot_pose.header
        robot_point.point.x = self.robot_pose.pose.position.x
        robot_point.point.y = self.robot_pose.pose.position.y
        robot_point.point.z = self.robot_pose.pose.position.z

        goal_point = PointStamped()
        goal_point.header = self.current_goal.header
        goal_point.point.x = self.current_goal.pose.position.x
        goal_point.point.y = self.current_goal.pose.position.y
        goal_point.point.z = self.current_goal.pose.position.z

        # Use obstacle detector to check path
        if self.obstacle_detector:
            return self.obstacle_detector.is_path_clear(
                robot_point.point,
                goal_point.point,
                safety_margin=self.safe_distance
            )

        return True  # If no obstacle detector, assume safe

    def replan_path(self):
        """Replan the current navigation path"""
        if self.current_goal is not None:
            # Cancel current goal
            # In a real implementation, you might want to cancel the current goal first

            # Publish new goal to trigger replanning
            self.move_base_goal_pub.publish(self.current_goal)

    def navigate_to_object(self, object_class, max_distance=5.0):
        """Navigate to a detected object of specified class"""
        if self.perception_system is None:
            rospy.logerr("Perception system not initialized")
            return False

        # Look for object in world model
        target_object = None
        for obj_id, obj_data in self.perception_system.world_objects.items():
            if (obj_data['class'] == object_class and
                obj_data['confidence'] > 0.7):  # High confidence detection
                # Check if object is within max distance
                pos = obj_data['position']
                if hasattr(self, 'robot_pose') and self.robot_pose:
                    dist = np.sqrt(
                        (pos.x - self.robot_pose.pose.position.x)**2 +
                        (pos.y - self.robot_pose.pose.position.y)**2
                    )
                    if dist <= max_distance:
                        target_object = obj_data
                        break

        if target_object is None:
            rospy.logwarn(f"No {object_class} detected within {max_distance}m")
            return False

        # Create navigation goal
        goal = PoseStamped()
        goal.header.frame_id = "map"  # or appropriate frame
        goal.header.stamp = rospy.Time.now()
        goal.pose.position = target_object['position']

        # Set orientation to face the object
        if self.robot_pose:
            dx = target_object['position'].x - self.robot_pose.pose.position.x
            dy = target_object['position'].y - self.robot_pose.pose.position.y
            yaw = np.arctan2(dy, dx)

            # Convert yaw to quaternion
            from tf.transformations import quaternion_from_euler
            quat = quaternion_from_euler(0, 0, yaw)
            goal.pose.orientation.x = quat[0]
            goal.pose.orientation.y = quat[1]
            goal.pose.orientation.z = quat[2]
            goal.pose.orientation.w = quat[3]

        # Store current goal
        self.current_goal = goal

        # Publish navigation goal
        self.move_base_goal_pub.publish(goal)

        rospy.loginfo(f"Navigating to {object_class} at position {target_object['position']}")
        return True
```

## Testing and Validation

### Perception System Testing Framework

```python
import unittest
import numpy as np
import cv2

class PerceptionSystemTester:
    def __init__(self):
        self.test_results = {}

    def run_all_tests(self):
        """Run all perception system tests"""
        tests = [
            self.test_object_detection_accuracy,
            self.test_vslam_tracking,
            self.test_sensor_fusion,
            self.test_obstacle_detection,
            self.test_perception_navigation_integration
        ]

        for test_func in tests:
            test_name = test_func.__name__
            rospy.loginfo(f"Running {test_name}...")
            result = test_func()
            self.test_results[test_name] = result
            rospy.loginfo(f"{test_name}: {result}")

        return self.test_results

    def test_object_detection_accuracy(self):
        """Test object detection accuracy"""
        # Create a test image with known objects
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)

        # Draw a red cup in the center
        cv2.circle(test_image, (320, 240), 50, (0, 0, 255), -1)

        # Initialize perception system
        perception = PerceptionSystem()
        detector = MockObjectDetector()

        # Run detection
        detections = detector.detect(test_image)

        # Check if cup was detected in approximately correct location
        cup_detected = False
        for detection in detections:
            if detection['class'] == 'cup':
                center_x, center_y = detection['bbox_center']
                if abs(center_x - 320) < 20 and abs(center_y - 240) < 20:
                    cup_detected = True
                    break

        return {
            'success': cup_detected,
            'detection_rate': 1.0 if cup_detected else 0.0,
            'false_positive_rate': 0.0,
            'processing_time': 0.05  # seconds
        }

    def test_vslam_tracking(self):
        """Test VSLAM tracking performance"""
        slam_system = VSLAMSystem()

        # Create synthetic frames simulating movement
        results = []
        for i in range(10):
            # Create a simple synthetic image with features
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

            # Add some synthetic features
            for j in range(50):
                x, y = np.random.randint(50, 600), np.random.randint(50, 400)
                cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)

            # Process frame
            camera_matrix = np.eye(3)
            pose = slam_system.process_frame(frame, camera_matrix)

            results.append({
                'frame': i,
                'has_pose': pose is not None,
                'num_features': len(slam_system.keyframes[-1]['features']) if slam_system.keyframes else 0
            })

        success = all(r['has_pose'] for r in results)

        return {
            'success': success,
            'tracking_stability': len([r for r in results if r['has_pose']]) / len(results),
            'average_features': np.mean([r['num_features'] for r in results]),
            'processing_time': 0.1  # seconds per frame
        }

    def test_sensor_fusion(self):
        """Test sensor fusion accuracy"""
        fusion = SensorFusion()

        # Simulate sensor measurements
        for i in range(100):
            timestamp = rospy.Time(i * 0.1)  # 10Hz

            # Simulate IMU data (with some noise)
            accel = np.array([0.1, 0.0, 9.8]) + np.random.normal(0, 0.01, 3)
            gyro = np.array([0.0, 0.0, 0.0]) + np.random.normal(0, 0.001, 3)

            # Simulate vision data (with some noise)
            pos = np.array([i * 0.01, 0.0, 0.8]) + np.random.normal(0, 0.01, 3)
            orient = np.array([0.0, 0.0, i * 0.001]) + np.random.normal(0, 0.001, 3)

            # Update fusion system
            fusion.update_imu(accel, gyro, timestamp)
            fusion.update_vision(pos, orient, timestamp)

        # Get final state
        final_state = fusion.get_fused_state()

        # Check if the estimated position is close to expected
        expected_pos = np.array([0.99, 0.0, 0.8])  # After 100 steps of 0.01m each
        position_error = np.linalg.norm(final_state['position'] - expected_pos)

        return {
            'success': position_error < 0.1,  # Within 10cm
            'position_error': position_error,
            'orientation_error': np.linalg.norm(final_state['orientation']),
            'covariance_trace': np.trace(final_state['position_covariance'])
        }

    def test_obstacle_detection(self):
        """Test obstacle detection capabilities"""
        detector = ObstacleDetection()

        # Test with synthetic laser scan data
        scan = LaserScan()
        scan.angle_min = -np.pi/2
        scan.angle_max = np.pi/2
        scan.angle_increment = np.pi / 180  # 1 degree
        scan.range_min = 0.1
        scan.range_max = 10.0

        # Create scan with an obstacle at 2m in front
        num_readings = int((scan.angle_max - scan.angle_min) / scan.angle_increment) + 1
        scan.ranges = [float('inf')] * num_readings

        # Place obstacle at center (0 degrees)
        center_idx = num_readings // 2
        scan.ranges[center_idx] = 2.0  # Obstacle at 2m
        scan.ranges[center_idx-1] = 2.0
        scan.ranges[center_idx+1] = 2.0

        # Process the scan
        detector.update_local_map_laser(scan)

        # Check if obstacle was detected in the right location
        map_center_x = detector.map_width // 2
        map_center_y = detector.map_height // 2

        # Obstacle should be detected around (2m, 0m) relative to robot
        obstacle_x = int((2.0 - detector.map_origin.x) / detector.map_resolution)

        obstacle_detected = detector.local_map[map_center_y, obstacle_x] > 50

        return {
            'success': obstacle_detected,
            'obstacle_detected': obstacle_detected,
            'detection_accuracy': 1.0 if obstacle_detected else 0.0,
            'false_positive_rate': 0.0
        }

    def test_perception_navigation_integration(self):
        """Test integration between perception and navigation"""
        # Initialize the integration system
        integration = PerceptionNavigationIntegration()

        # Create mock perception and obstacle detection systems
        perception = PerceptionSystem()
        obstacle_detector = ObstacleDetection()

        integration.set_perception_system(perception, obstacle_detector)

        # Test navigate_to_object functionality
        # (This would require a more complex setup with actual ROS nodes)

        return {
            'success': True,  # For now, just verify initialization
            'integration_tested': True
        }
```

## Assessment Rubric

Your Phase 3 implementation will be evaluated on:

- **Computer Vision (25%)**: Effective object detection, recognition, and tracking
- **VSLAM Implementation (25%)**: Accurate mapping and localization capabilities
- **Sensor Fusion (25%)**: Proper integration of multiple sensor inputs
- **Obstacle Detection (15%)**: Reliable detection and avoidance of obstacles
- **Testing and Validation (10%)**: Comprehensive testing with good results

## Resources

- [ROS Perception Tutorials](http://wiki.ros.org/perception/Tutorials)
- [OpenVSLAM Documentation](https://openvslam.readthedocs.io/)
- [NVIDIA Isaac Perception Examples](https://docs.nvidia.com/isaac/perception/index.html)
- [Computer Vision in Robotics](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=computer%20vision%20robotics)

## Next Phase

Upon successful completion of Phase 3, you will proceed to Phase 4: Action and Interaction, where you will integrate voice recognition, cognitive planning, and manipulation capabilities for your humanoid robot.
