# Chapter 2: Isaac ROS Pipelines

## Overview
This chapter explores the Isaac ROS (Robot Operating System) framework, focusing on hardware-accelerated perception and navigation pipelines. Students will learn to leverage NVIDIA's GPU acceleration for computer vision, SLAM, and navigation tasks, creating efficient perception-action loops for humanoid robots operating in complex environments.

## Learning Objectives
By the end of this chapter, students will be able to:
- Configure and deploy Isaac ROS perception pipelines with GPU acceleration
- Implement hardware-accelerated VSLAM (Visual SLAM) for humanoid navigation
- Create efficient perception-action loops using Isaac ROS extensions
- Integrate Isaac ROS with navigation and planning systems
- Optimize perception pipelines for real-time humanoid robot applications
- Understand the integration between Isaac Sim and Isaac ROS for sim-to-real transfer

## 1. Introduction to Isaac ROS

Isaac ROS is NVIDIA's accelerated perception and navigation package for ROS 2, designed to harness the power of NVIDIA GPUs for robotics applications. For humanoid robots, Isaac ROS provides:

- **Hardware-accelerated perception**: GPU-powered computer vision algorithms
- **Real-time SLAM**: Visual and LiDAR-based simultaneous localization and mapping
- **Deep learning inference**: Optimized neural network execution on GPU
- **Sensor processing**: Accelerated processing of camera, LiDAR, and IMU data
- **Navigation integration**: Hardware-accelerated path planning and navigation

### Key Isaac ROS Components for Humanoid Robotics:
- **Isaac ROS Image Pipeline**: Accelerated image preprocessing and rectification
- **Isaac ROS Visual SLAM**: GPU-accelerated visual odometry and mapping
- **Isaac ROS Detection Pipeline**: Real-time object detection and tracking
- **Isaac ROS Manipulation**: GPU-accelerated grasp planning and manipulation
- **Isaac ROS Navigation**: Hardware-accelerated navigation and path planning

## 2. Installing Isaac ROS

### Prerequisites:
```bash
# Ensure NVIDIA GPU drivers are installed
nvidia-smi

# Install ROS 2 Iron (or appropriate version)
sudo apt update
sudo apt install ros-iron-desktop

# Install Isaac ROS dependencies
sudo apt install ros-iron-isaac-ros-* ros-iron-isaac-ros-gems
```

### Isaac ROS Installation:
```bash
# Install Isaac ROS packages via apt
sudo apt install ros-iron-isaac-ros-all

# Or build from source for latest features
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam.git
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_bi3d.git
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_detectnet.git
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation.git
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_apriltag.git
```

### Verify Installation:
```bash
# Check Isaac ROS packages
ros2 pkg list | grep isaac_ros

# Run Isaac ROS diagnostic
ros2 run isaac_ros_common isaac_ros_diagnostic
```

## 3. Isaac ROS Image Pipeline

### Image Rectification and Preprocessing:
```python
# image_pipeline.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from stereo_msgs.msg import DisparityImage
from cv_bridge import CvBridge
import numpy as np
import cv2

class IsaacImagePipeline(Node):
    def __init__(self):
        super().__init__('isaac_image_pipeline')

        # Initialize CV Bridge
        self.bridge = CvBridge()

        # Image and camera info subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.image_callback, 10
        )
        self.info_sub = self.create_subscription(
            CameraInfo, '/camera/rgb/camera_info', self.info_callback, 10
        )

        # Publishers for processed images
        self.rectified_pub = self.create_publisher(Image, '/camera/rgb/image_rect', 10)
        self.disparity_pub = self.create_publisher(DisparityImage, '/camera/disparity', 10)

        # Camera parameters storage
        self.camera_matrix = None
        self.dist_coeffs = None

        # Isaac ROS image processing components
        self.rectification_map = None
        self.initialized = False

        self.get_logger().info('Isaac Image Pipeline initialized')

    def info_callback(self, msg):
        """Process camera calibration information"""
        if not self.initialized:
            # Extract camera parameters
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d)

            # Create rectification map using OpenCV (Isaac ROS would use CUDA)
            self.rectification_map = self.create_rectification_maps(
                self.camera_matrix, self.dist_coeffs, (msg.width, msg.height)
            )
            self.initialized = True

    def create_rectification_maps(self, camera_matrix, dist_coeffs, image_size):
        """Create rectification maps for undistortion"""
        # Create new camera matrix for undistorted image
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, image_size, 1, image_size
        )

        # Generate rectification maps
        mapx, mapy = cv2.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, None, new_camera_matrix, image_size, cv2.CV_32FC1
        )

        return mapx, mapy, new_camera_matrix

    def image_callback(self, msg):
        """Process incoming image with Isaac ROS pipeline"""
        if not self.initialized:
            return

        # Convert ROS image to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Apply rectification (Isaac ROS would use GPU acceleration)
        mapx, mapy, _ = self.rectification_map
        rectified_image = cv2.remap(cv_image, mapx, mapy, cv2.INTER_LINEAR)

        # Convert back to ROS message
        rectified_msg = self.bridge.cv2_to_imgmsg(rectified_image, encoding='bgr8')
        rectified_msg.header = msg.header
        self.rectified_pub.publish(rectified_msg)

        # Process image for further analysis
        self.process_image_features(rectified_image, msg.header)

    def process_image_features(self, image, header):
        """Extract features using Isaac ROS optimized algorithms"""
        # This would typically use Isaac ROS's GPU-accelerated feature extraction
        # For demonstration, we'll use a simple approach

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply edge detection (Isaac ROS would use optimized kernels)
        edges = cv2.Canny(gray, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Publish features or use for navigation
        self.publish_features(contours, header)

    def publish_features(self, contours, header):
        """Publish extracted features for downstream processing"""
        # In a real Isaac ROS pipeline, this would publish to specialized topics
        # like /features or connect to SLAM nodes
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            # Extract bounding box or other descriptors
            x, y, w, h = cv2.boundingRect(largest_contour)

            # For now, just log the feature
            self.get_logger().info(f'Detected feature: {w}x{h} at ({x},{y})')

def main(args=None):
    rclpy.init(args=args)
    pipeline = IsaacImagePipeline()

    try:
        rclpy.spin(pipeline)
    except KeyboardInterrupt:
        pipeline.get_logger().info('Shutting down Isaac Image Pipeline')
    finally:
        pipeline.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 4. Isaac ROS Visual SLAM

### Visual SLAM Node Implementation:
```python
# visual_slam.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, PointStamped
from visualization_msgs.msg import MarkerArray, Marker
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
import numpy as np
import cv2
from collections import deque

class IsaacVisualSLAM(Node):
    def __init__(self):
        super().__init__('isaac_visual_slam')

        # Initialize components
        self.bridge = CvBridge()
        self.br = TransformBroadcaster(self)

        # Store camera parameters
        self.camera_info = None
        self.imu_data = deque(maxlen=10)

        # SLAM state
        self.current_pose = np.eye(4)  # 4x4 transformation matrix
        self.keyframes = []
        self.map_points = []
        self.tracking_initialized = False

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.image_callback, 10
        )
        self.info_sub = self.create_subscription(
            CameraInfo, '/camera/rgb/camera_info', self.info_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )

        # Publishers
        self.odom_pub = self.create_publisher(Odometry, '/visual_odom', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/visual_pose', 10)
        self.map_pub = self.create_publisher(MarkerArray, '/visual_map', 10)
        self.keyframe_pub = self.create_publisher(MarkerArray, '/keyframes', 10)

        # Feature tracking parameters
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.01,
            minDistance=20,
            blockSize=7
        )
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        # Previous frame for optical flow
        self.prev_gray = None
        self.prev_features = None

        self.get_logger().info('Isaac Visual SLAM initialized')

    def info_callback(self, msg):
        """Store camera calibration information"""
        self.camera_info = msg

    def imu_callback(self, msg):
        """Store IMU data for sensor fusion"""
        self.imu_data.append({
            'orientation': [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
            'angular_velocity': [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
            'linear_acceleration': [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z],
            'timestamp': msg.header.stamp
        })

    def image_callback(self, msg):
        """Process image for visual SLAM"""
        # Convert to grayscale
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')

        if self.prev_gray is None:
            # Initialize feature tracking
            self.prev_gray = cv_image
            self.prev_features = cv2.goodFeaturesToTrack(
                cv_image, mask=None, **self.feature_params
            )
            if self.prev_features is not None:
                self.tracking_initialized = True
            return

        # Calculate optical flow
        if self.prev_features is not None:
            curr_features, status, error = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, cv_image, self.prev_features, None, **self.lk_params
            )

            # Filter good points
            good_new = curr_features[status == 1]
            good_old = self.prev_features[status == 1]

            if len(good_new) > 10:  # Require minimum features for tracking
                # Estimate motion using essential matrix
                E, mask = cv2.findEssentialMat(
                    good_new, good_old,
                    focal=self.camera_info.k[0],  # fx
                    pp=(self.camera_info.k[2], self.camera_info.k[5]),  # cx, cy
                    method=cv2.RANSAC, threshold=1.0
                )

                if E is not None:
                    # Recover pose from essential matrix
                    _, R, t, _ = cv2.recoverPose(E, good_new, good_old)

                    # Update pose (simplified - real SLAM would be more sophisticated)
                    delta_transform = np.eye(4)
                    delta_transform[:3, :3] = R
                    delta_transform[:3, 3] = t.flatten() * 0.1  # Scale factor

                    self.current_pose = self.current_pose @ delta_transform

                    # Publish odometry
                    self.publish_odometry(msg.header.stamp, self.current_pose)

                    # Add keyframe if movement is significant
                    if self.should_add_keyframe(delta_transform):
                        self.add_keyframe(msg, self.current_pose)

            # Update for next iteration
            self.prev_gray = cv_image.copy()
            if len(good_new) > 20:  # Maintain minimum features
                self.prev_features = good_new.reshape(-1, 1, 2)
            else:
                # Re-detect features
                self.prev_features = cv2.goodFeaturesToTrack(
                    cv_image, mask=None, **self.feature_params
                )
        else:
            # Re-detect features if lost
            self.prev_gray = cv_image
            self.prev_features = cv2.goodFeaturesToTrack(
                cv_image, mask=None, **self.feature_params
            )

    def should_add_keyframe(self, delta_transform):
        """Determine if a keyframe should be added"""
        # Check translation magnitude
        translation_norm = np.linalg.norm(delta_transform[:3, 3])

        # Check rotation magnitude
        rotation_matrix = delta_transform[:3, :3]
        angle = np.arccos(np.clip((np.trace(rotation_matrix) - 1) / 2, -1, 1))

        # Add keyframe if movement is significant
        return translation_norm > 0.1 or angle > 0.1  # 10cm or ~5.7 degrees

    def add_keyframe(self, image_msg, pose):
        """Add a keyframe to the map"""
        keyframe = {
            'image': image_msg,
            'pose': pose.copy(),
            'timestamp': image_msg.header.stamp,
            'features': self.prev_features
        }
        self.keyframes.append(keyframe)

        # Limit keyframes to prevent memory growth
        if len(self.keyframes) > 100:
            self.keyframes.pop(0)

        # Publish keyframe visualization
        self.publish_keyframes()

    def publish_odometry(self, stamp, pose):
        """Publish odometry information"""
        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'

        # Set position
        odom_msg.pose.pose.position.x = pose[0, 3]
        odom_msg.pose.pose.position.y = pose[1, 3]
        odom_msg.pose.pose.position.z = pose[2, 3]

        # Convert rotation matrix to quaternion
        rotation_matrix = pose[:3, :3]
        qw, qx, qy, qz = self.rotation_matrix_to_quaternion(rotation_matrix)
        odom_msg.pose.pose.orientation.w = qw
        odom_msg.pose.pose.orientation.x = qx
        odom_msg.pose.pose.orientation.y = qy
        odom_msg.pose.pose.orientation.z = qz

        # Set velocities (approximate)
        # In real implementation, these would come from differentiation

        self.odom_pub.publish(odom_msg)

        # Publish pose
        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = 'odom'
        pose_msg.pose = odom_msg.pose.pose
        self.pose_pub.publish(pose_msg)

        # Broadcast transform
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = pose[0, 3]
        t.transform.translation.y = pose[1, 3]
        t.transform.translation.z = pose[2, 3]
        t.transform.rotation.w = qw
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        self.br.sendTransform(t)

    def rotation_matrix_to_quaternion(self, R):
        """Convert rotation matrix to quaternion"""
        trace = np.trace(R)
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s

        return qw, qx, qy, qz

    def publish_keyframes(self):
        """Publish keyframe visualization markers"""
        marker_array = MarkerArray()

        for i, keyframe in enumerate(self.keyframes):
            marker = Marker()
            marker.header.frame_id = "odom"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "keyframes"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            # Position from keyframe pose
            marker.pose.position.x = keyframe['pose'][0, 3]
            marker.pose.position.y = keyframe['pose'][1, 3]
            marker.pose.position.z = keyframe['pose'][2, 3]

            # Orientation from keyframe pose
            R = keyframe['pose'][:3, :3]
            qw, qx, qy, qz = self.rotation_matrix_to_quaternion(R)
            marker.pose.orientation.w = qw
            marker.pose.orientation.x = qx
            marker.pose.orientation.y = qy
            marker.pose.orientation.z = qz

            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0

            marker_array.markers.append(marker)

        self.keyframe_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    slam = IsaacVisualSLAM()

    try:
        rclpy.spin(slam)
    except KeyboardInterrupt:
        slam.get_logger().info('Shutting down Isaac Visual SLAM')
    finally:
        slam.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 5. Isaac ROS Deep Learning Integration

### GPU-Accelerated Neural Network Inference:
```python
# deep_learning_pipeline.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose
from geometry_msgs.msg import Point
from std_msgs.msg import Header
from cv_bridge import CvBridge
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image as PILImage

class IsaacDeepLearningPipeline(Node):
    def __init__(self):
        super().__init__('isaac_deep_learning_pipeline')

        # Initialize components
        self.bridge = CvBridge()

        # Load pre-trained models (Isaac ROS provides optimized models)
        self.detection_model = self.load_detection_model()
        self.segmentation_model = self.load_segmentation_model()
        self.depth_model = self.load_depth_model()

        # Model parameters
        self.input_size = (640, 480)  # Input size for models
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.image_callback, 10
        )

        # Publishers
        self.detections_pub = self.create_publisher(Detection2DArray, '/detections', 10)
        self.segmentation_pub = self.create_publisher(Image, '/segmentation', 10)
        self.depth_pub = self.create_publisher(Image, '/depth_estimate', 10)

        # Transformation for neural network input
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.get_logger().info('Isaac Deep Learning Pipeline initialized')

    def load_detection_model(self):
        """Load object detection model (YOLO, DetectNet, etc.)"""
        # In Isaac ROS, this would load an optimized TensorRT model
        # For demonstration, we'll create a dummy model
        try:
            # Load a pre-trained model - in Isaac ROS this would be optimized for Jetson
            import torchvision.models as models
            model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            model.eval()

            # Move to GPU if available
            if torch.cuda.is_available():
                model = model.cuda()
                self.get_logger().info('Detection model loaded on GPU')
            else:
                self.get_logger().info('Detection model loaded on CPU')

            return model
        except Exception as e:
            self.get_logger().error(f'Failed to load detection model: {e}')
            return None

    def load_segmentation_model(self):
        """Load semantic segmentation model"""
        try:
            import torchvision.models as models
            model = models.segmentation.deeplabv3_resnet50(pretrained=True)
            model.eval()

            if torch.cuda.is_available():
                model = model.cuda()
                self.get_logger().info('Segmentation model loaded on GPU')
            else:
                self.get_logger().info('Segmentation model loaded on CPU')

            return model
        except Exception as e:
            self.get_logger().error(f'Failed to load segmentation model: {e}')
            return None

    def load_depth_model(self):
        """Load monocular depth estimation model"""
        try:
            # For monocular depth estimation
            # In practice, would use MiDaS or similar
            return None  # Placeholder
        except Exception as e:
            self.get_logger().error(f'Failed to load depth model: {e}')
            return None

    def image_callback(self, msg):
        """Process image through deep learning pipeline"""
        # Convert ROS image to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Process through each model
        detections = self.run_object_detection(cv_image)
        segmentation = self.run_segmentation(cv_image)
        depth = self.run_depth_estimation(cv_image)

        # Publish results
        if detections is not None:
            self.publish_detections(detections, msg.header)

        if segmentation is not None:
            seg_msg = self.bridge.cv2_to_imgmsg(segmentation, encoding='mono8')
            seg_msg.header = msg.header
            self.segmentation_pub.publish(seg_msg)

    def run_object_detection(self, image):
        """Run object detection on image"""
        if self.detection_model is None:
            return None

        # Preprocess image
        pil_image = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        input_tensor = self.transform(pil_image).unsqueeze(0)

        # Move to GPU if available
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()

        try:
            with torch.no_grad():
                predictions = self.detection_model(input_tensor)

            # Process predictions
            pred = predictions[0]
            boxes = pred['boxes'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()

            # Filter by confidence
            valid_indices = scores > self.confidence_threshold
            valid_boxes = boxes[valid_indices]
            valid_labels = labels[valid_indices]
            valid_scores = scores[valid_indices]

            # Apply NMS (Non-Maximum Suppression)
            indices = cv2.dnn.NMSBoxes(
                bboxes=[[int(box[0]), int(box[1]), int(box[2]-box[0]), int(box[3]-box[1])] for box in valid_boxes],
                scores=valid_scores.astype(float),
                score_threshold=self.confidence_threshold,
                nms_threshold=self.nms_threshold
            )

            if len(indices) > 0:
                final_boxes = valid_boxes[indices.flatten()]
                final_labels = valid_labels[indices.flatten()]
                final_scores = valid_scores[indices.flatten()]

                return {
                    'boxes': final_boxes,
                    'labels': final_labels,
                    'scores': final_scores
                }

        except Exception as e:
            self.get_logger().error(f'Detection inference failed: {e}')

        return None

    def run_segmentation(self, image):
        """Run semantic segmentation on image"""
        if self.segmentation_model is None:
            return None

        try:
            # Preprocess image
            pil_image = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            input_tensor = self.transform(pil_image).unsqueeze(0)

            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()

            with torch.no_grad():
                output = self.segmentation_model(input_tensor)['out']
                output_predictions = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

            # Convert to color segmentation map for visualization
            segmentation_map = self.colorize_segmentation(output_predictions)
            return segmentation_map

        except Exception as e:
            self.get_logger().error(f'Segmentation inference failed: {e}')
            return None

    def colorize_segmentation(self, segmentation_map):
        """Convert segmentation map to color image for visualization"""
        # This is a simplified colorization - real implementation would use proper colormap
        colored_map = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8)

        # Create a simple color mapping (in practice, use proper COCO or Cityscapes colors)
        unique_labels = np.unique(segmentation_map)
        colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                  for _ in unique_labels]

        for i, label in enumerate(unique_labels):
            mask = segmentation_map == label
            colored_map[mask] = colors[i]

        return colored_map

    def run_depth_estimation(self, image):
        """Run monocular depth estimation"""
        # Placeholder for depth estimation
        # In Isaac ROS, this would use optimized depth models
        return None

    def publish_detections(self, detections, header):
        """Publish detection results"""
        detections_msg = Detection2DArray()
        detections_msg.header = header

        for i, (box, label, score) in enumerate(zip(detections['boxes'],
                                                   detections['labels'],
                                                   detections['scores'])):
            detection = Detection2D()
            detection.header = header
            detection.id = str(i)

            # Set bounding box
            bbox = detection.bbox
            bbox.center.x = float((box[0] + box[2]) / 2)
            bbox.center.y = float((box[1] + box[3]) / 2)
            bbox.size_x = float(box[2] - box[0])
            bbox.size_y = float(box[3] - box[1])

            # Set hypothesis
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.id = str(label)
            hypothesis.score = float(score)
            detection.results.append(hypothesis)

            detections_msg.detections.append(detection)

        self.detections_pub.publish(detections_msg)

def main(args=None):
    rclpy.init(args=args)
    dl_pipeline = IsaacDeepLearningPipeline()

    try:
        rclpy.spin(dl_pipeline)
    except KeyboardInterrupt:
        dl_pipeline.get_logger().info('Shutting down Isaac Deep Learning Pipeline')
    finally:
        dl_pipeline.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 6. Isaac ROS Navigation Integration

### Hardware-Accelerated Navigation:
```python
# navigation_pipeline.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from sensor_msgs.msg import LaserScan, PointCloud2
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray
import numpy as np
import cv2
from scipy.spatial import KDTree
import heapq

class IsaacNavigationPipeline(Node):
    def __init__(self):
        super().__init__('isaac_navigation_pipeline')

        # Navigation state
        self.current_pose = None
        self.map = None
        self.path = []
        self.goal = None
        self.obstacles = []
        self.waypoints = []

        # Configuration
        self.resolution = 0.05  # meters per pixel
        self.robot_radius = 0.3  # meters
        self.inflation_radius = 0.5  # meters for obstacle inflation

        # Subscribers
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/initialpose', self.pose_callback, 10
        )
        self.goal_sub = self.create_subscription(
            PoseStamped, '/move_base_simple/goal', self.goal_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/visual_odom', self.odom_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10
        )

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, '/nav_path', 10)
        self.local_plan_pub = self.create_publisher(Path, '/local_plan', 10)
        self.global_plan_pub = self.create_publisher(Path, '/global_plan', 10)

        # Timers
        self.navigation_timer = self.create_timer(0.1, self.navigation_callback)
        self.control_timer = self.create_timer(0.05, self.control_callback)  # 20Hz control

        self.get_logger().info('Isaac Navigation Pipeline initialized')

    def map_callback(self, msg):
        """Process occupancy grid map"""
        self.map = {
            'data': np.array(msg.data).reshape(msg.info.height, msg.info.width),
            'info': msg.info,
            'resolution': msg.info.resolution
        }
        self.get_logger().info(f'Map received: {msg.info.width}x{msg.info.height}')

    def pose_callback(self, msg):
        """Update initial pose"""
        self.current_pose = {
            'x': msg.pose.pose.position.x,
            'y': msg.pose.pose.position.y,
            'theta': self.quaternion_to_yaw(msg.pose.pose.orientation)
        }

    def odom_callback(self, msg):
        """Update current pose from odometry"""
        self.current_pose = {
            'x': msg.pose.pose.position.x,
            'y': msg.pose.pose.position.y,
            'theta': self.quaternion_to_yaw(msg.pose.pose.orientation)
        }

    def goal_callback(self, msg):
        """Set navigation goal"""
        self.goal = {
            'x': msg.pose.position.x,
            'y': msg.pose.position.y
        }
        self.get_logger().info(f'New goal set: ({self.goal["x"]}, {self.goal["y"]})')

        # Plan path to goal
        if self.map and self.current_pose:
            self.plan_path()

    def scan_callback(self, msg):
        """Process laser scan for local obstacle detection"""
        # Convert laser scan to obstacle points
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        valid_ranges = np.array(msg.ranges)
        valid_ranges[valid_ranges > msg.range_max] = np.inf
        valid_ranges[valid_ranges < msg.range_min] = np.inf

        # Get obstacle positions in robot frame
        x_points = valid_ranges * np.cos(angles)
        y_points = valid_ranges * np.sin(angles)

        # Filter valid points
        valid_mask = np.isfinite(valid_ranges)
        self.obstacles = np.column_stack((x_points[valid_mask], y_points[valid_mask]))

    def plan_path(self):
        """Plan global path using A* algorithm"""
        if not self.map or not self.current_pose or not self.goal:
            return

        # Convert world coordinates to grid coordinates
        start_grid = self.world_to_grid(self.current_pose['x'], self.current_pose['y'])
        goal_grid = self.world_to_grid(self.goal['x'], self.goal['y'])

        if not self.is_valid_cell(start_grid[0], start_grid[1]) or \
           not self.is_valid_cell(goal_grid[0], goal_grid[1]):
            self.get_logger().warn('Start or goal position is in obstacle space')
            return

        # Run A* path planning
        path = self.a_star_planning(start_grid, goal_grid)

        if path:
            # Convert grid path back to world coordinates
            world_path = []
            for grid_x, grid_y in path:
                world_x, world_y = self.grid_to_world(grid_x, grid_y)
                world_path.append((world_x, world_y))

            self.path = world_path
            self.publish_path(world_path)
        else:
            self.get_logger().warn('No valid path found')

    def a_star_planning(self, start, goal):
        """A* path planning algorithm"""
        # Heuristic function
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        # Possible movements (8-connected)
        moves = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

        # Priority queue: (cost, x, y)
        open_set = [(0, start[0], start[1])]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}

        while open_set:
            current_cost, x, y = heapq.heappop(open_set)

            if (x, y) == goal:
                # Reconstruct path
                path = []
                current = (x, y)
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            for dx, dy in moves:
                neighbor = (x + dx, y + dy)

                # Check if neighbor is valid
                if not self.is_valid_cell(neighbor[0], neighbor[1]):
                    continue

                # Calculate tentative g_score
                if abs(dx) + abs(dy) == 2:  # Diagonal move
                    tentative_g = g_score[(x, y)] + np.sqrt(2)
                else:  # Horizontal/vertical move
                    tentative_g = g_score[(x, y)] + 1

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # This path to neighbor is better
                    came_from[neighbor] = (x, y)
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor[0], neighbor[1]))

        return None  # No path found

    def is_valid_cell(self, x, y):
        """Check if grid cell is valid (not occupied)"""
        if not self.map:
            return False

        if x < 0 or x >= self.map['info'].width or y < 0 or y >= self.map['info'].height:
            return False

        # Check occupancy grid value
        idx = y * self.map['info'].width + x
        if idx >= len(self.map['data']):
            return False

        occupancy_value = self.map['data'][idx]
        return occupancy_value < 50  # Threshold for free space (0-100 scale)

    def world_to_grid(self, x, y):
        """Convert world coordinates to grid coordinates"""
        if not self.map:
            return (0, 0)

        grid_x = int((x - self.map['info'].origin.position.x) / self.map['info'].resolution)
        grid_y = int((y - self.map['info'].origin.position.y) / self.map['info'].resolution)

        return (grid_x, grid_y)

    def grid_to_world(self, x, y):
        """Convert grid coordinates to world coordinates"""
        if not self.map:
            return (0, 0)

        world_x = x * self.map['info'].resolution + self.map['info'].origin.position.x
        world_y = y * self.map['info'].resolution + self.map['info'].origin.position.y

        return (world_x, world_y)

    def navigation_callback(self):
        """Main navigation logic"""
        if not self.current_pose or not self.path:
            return

        # Update local path based on current position
        self.update_local_path()

    def update_local_path(self):
        """Update local path considering current position and obstacles"""
        if not self.path or not self.current_pose:
            return

        # Find closest point on path
        current_pos = np.array([self.current_pose['x'], self.current_pose['y']])
        path_array = np.array(self.path)

        if len(path_array) == 0:
            return

        # Calculate distances to all path points
        distances = np.linalg.norm(path_array - current_pos, axis=1)
        closest_idx = np.argmin(distances)

        # Take next N points for local planning
        look_ahead = 10
        end_idx = min(closest_idx + look_ahead, len(self.path))
        local_path = self.path[closest_idx:end_idx]

        if local_path:
            self.publish_local_path(local_path)

    def control_callback(self):
        """Send velocity commands to robot"""
        if not self.current_pose or not self.path:
            # Stop robot if no path
            cmd = Twist()
            self.cmd_vel_pub.publish(cmd)
            return

        # Calculate control commands (simplified pure pursuit)
        cmd = self.pure_pursuit_control()
        self.cmd_vel_pub.publish(cmd)

    def pure_pursuit_control(self):
        """Pure pursuit path following controller"""
        cmd = Twist()

        if not self.path or len(self.path) < 2:
            return cmd

        # Find look-ahead point
        current_pos = np.array([self.current_pose['x'], self.current_pose['y']])
        lookahead_dist = 0.5  # meters

        # Search for look-ahead point
        look_ahead_point = None
        for point in self.path:
            point_array = np.array([point[0], point[1]])
            dist = np.linalg.norm(point_array - current_pos)
            if dist >= lookahead_dist:
                look_ahead_point = point
                break

        if look_ahead_point is not None:
            # Calculate heading error
            dx = look_ahead_point[0] - self.current_pose['x']
            dy = look_ahead_point[1] - self.current_pose['y']
            target_angle = np.arctan2(dy, dx)

            # Calculate angle error
            angle_error = target_angle - self.current_pose['theta']
            # Normalize angle error to [-pi, pi]
            while angle_error > np.pi:
                angle_error -= 2 * np.pi
            while angle_error < -np.pi:
                angle_error += 2 * np.pi

            # Simple proportional controller
            cmd.linear.x = min(0.5, max(0.1, 0.5 * np.cos(angle_error)))  # Forward speed
            cmd.angular.z = 1.0 * angle_error  # Angular speed

        return cmd

    def publish_path(self, path_points):
        """Publish global path"""
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'

        for x, y in path_points:
            pose = PoseStamped()
            pose.header.stamp = path_msg.header.stamp
            pose.header.frame_id = 'map'
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self.global_plan_pub.publish(path_msg)

    def publish_local_path(self, path_points):
        """Publish local path"""
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'

        for x, y in path_points:
            pose = PoseStamped()
            pose.header.stamp = path_msg.header.stamp
            pose.header.frame_id = 'map'
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self.local_plan_pub.publish(path_msg)

    def quaternion_to_yaw(self, orientation):
        """Convert quaternion to yaw angle"""
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return yaw

def main(args=None):
    rclpy.init(args=args)
    nav_pipeline = IsaacNavigationPipeline()

    try:
        rclpy.spin(nav_pipeline)
    except KeyboardInterrupt:
        nav_pipeline.get_logger().info('Shutting down Isaac Navigation Pipeline')
    finally:
        nav_pipeline.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 7. Isaac ROS for Humanoid Manipulation

### Manipulation Pipeline:
```python
# manipulation_pipeline.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import PoseStamped, PointStamped, Vector3
from std_msgs.msg import String, Float32
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformListener, Buffer
from cv_bridge import CvBridge
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

class IsaacManipulationPipeline(Node):
    def __init__(self):
        super().__init__('isaac_manipulation_pipeline')

        # Initialize components
        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Robot state
        self.joint_states = JointState()
        self.end_effector_pose = None
        self.camera_pose = None

        # Manipulation state
        self.target_object = None
        self.grasp_pose = None
        self.planned_trajectory = []

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )
        self.camera_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.camera_callback, 10
        )
        self.object_sub = self.create_subscription(
            PointStamped, '/target_object', self.object_callback, 10
        )

        # Publishers
        self.joint_cmd_pub = self.create_publisher(JointState, '/joint_commands', 10)
        self.grasp_pose_pub = self.create_publisher(PoseStamped, '/grasp_pose', 10)
        self.trajectory_pub = self.create_publisher(JointState, '/trajectory', 10)
        self.visualization_pub = self.create_publisher(MarkerArray, '/manipulation_viz', 10)

        # Services for manipulation
        self.grasp_service = self.create_service(
            String, '/perform_grasp', self.grasp_callback
        )
        self.pick_place_service = self.create_service(
            String, '/pick_and_place', self.pick_place_callback
        )

        # Manipulation parameters
        self.reach_threshold = 0.1  # meters
        self.approach_distance = 0.15  # meters to approach object

        self.get_logger().info('Isaac Manipulation Pipeline initialized')

    def joint_state_callback(self, msg):
        """Update joint state information"""
        self.joint_states = msg

    def camera_callback(self, msg):
        """Process camera image for object detection"""
        # In Isaac ROS, this would run object detection models
        # For now, we'll just convert the image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Process image for object detection
        # This would typically use Isaac ROS perception nodes
        detected_objects = self.detect_objects(cv_image)

        if detected_objects:
            # Publish detected objects for further processing
            self.process_detected_objects(detected_objects, msg.header)

    def detect_objects(self, image):
        """Detect objects in image using Isaac ROS perception"""
        # This would use Isaac ROS's optimized detection models
        # For demonstration, we'll return mock objects
        # In real implementation, this would run neural networks on GPU
        return []

    def object_callback(self, msg):
        """Process target object information"""
        self.target_object = {
            'x': msg.point.x,
            'y': msg.point.y,
            'z': msg.point.z,
            'frame_id': msg.header.frame_id
        }

        # Plan grasp for the target object
        self.plan_grasp()

    def plan_grasp(self):
        """Plan grasp pose for target object"""
        if not self.target_object or not self.end_effector_pose:
            return

        # Calculate approach position (slightly away from object)
        approach_pos = np.array([
            self.target_object['x'],
            self.target_object['y'],
            self.target_object['z']
        ])

        # Calculate approach direction (from object to gripper)
        ee_pos = np.array([
            self.end_effector_pose.pose.position.x,
            self.end_effector_pose.pose.position.y,
            self.end_effector_pose.pose.position.z
        ])

        # Direction from object to end-effector
        approach_dir = (ee_pos - approach_pos) / np.linalg.norm(ee_pos - approach_pos)

        # Move approach position back by approach distance
        approach_pos = approach_pos + approach_dir * self.approach_distance

        # Create grasp pose
        grasp_pose = PoseStamped()
        grasp_pose.header.frame_id = self.target_object['frame_id']
        grasp_pose.header.stamp = self.get_clock().now().to_msg()
        grasp_pose.pose.position.x = approach_pos[0]
        grasp_pose.pose.position.y = approach_pos[1]
        grasp_pose.pose.position.z = approach_pos[2]

        # Set orientation (typically pointing down toward object)
        # This would be more sophisticated in real implementation
        grasp_pose.pose.orientation.w = 1.0
        grasp_pose.pose.orientation.x = 0.0
        grasp_pose.pose.orientation.y = 0.0
        grasp_pose.pose.orientation.z = 0.0

        self.grasp_pose = grasp_pose
        self.grasp_pose_pub.publish(grasp_pose)

        # Visualize grasp pose
        self.visualize_grasp_pose(grasp_pose)

    def process_detected_objects(self, objects, header):
        """Process detected objects for manipulation"""
        # This would filter objects based on manipulation goals
        # For now, we'll just log detected objects
        for obj in objects:
            self.get_logger().info(f'Detected object: {obj}')

    def grasp_callback(self, request, response):
        """Perform grasp action"""
        if self.grasp_pose:
            # Move to grasp pose
            success = self.move_to_grasp_pose()
            if success:
                # Close gripper
                self.close_gripper()
                response.data = 'Grasp successful'
            else:
                response.data = 'Grasp failed'
        else:
            response.data = 'No grasp pose available'

        return response

    def pick_place_callback(self, request, response):
        """Perform pick and place action"""
        if self.grasp_pose:
            # Move to grasp pose
            if self.move_to_grasp_pose():
                # Close gripper
                self.close_gripper()

                # Lift object
                self.lift_object()

                # Move to place location
                place_location = self.get_place_location()
                if self.move_to_place_location(place_location):
                    # Open gripper
                    self.open_gripper()

                    # Retreat
                    self.retreat_after_placement()

                    response.data = 'Pick and place successful'
                else:
                    response.data = 'Move to place location failed'
            else:
                response.data = 'Grasp failed'
        else:
            response.data = 'No grasp pose available'

        return response

    def move_to_grasp_pose(self):
        """Move end effector to grasp pose"""
        # This would use inverse kinematics to plan joint trajectory
        # For demonstration, we'll just return success
        self.get_logger().info('Moving to grasp pose')
        return True

    def close_gripper(self):
        """Close robot gripper"""
        self.get_logger().info('Closing gripper')
        # Publish gripper command

    def lift_object(self):
        """Lift object after grasping"""
        self.get_logger().info('Lifting object')
        # Move upward to lift object

    def get_place_location(self):
        """Get location to place object"""
        # This would be determined by task or user input
        # For now, return a fixed location
        return {'x': 0.5, 'y': 0.0, 'z': 0.5}

    def move_to_place_location(self, location):
        """Move to place location"""
        self.get_logger().info(f'Moving to place location: {location}')
        # Move to specified location
        return True

    def open_gripper(self):
        """Open robot gripper"""
        self.get_logger().info('Opening gripper')
        # Publish gripper command

    def retreat_after_placement(self):
        """Retreat after placing object"""
        self.get_logger().info('Retreating after placement')
        # Move away from placed object

    def visualize_grasp_pose(self, grasp_pose):
        """Visualize grasp pose in RViz"""
        marker_array = MarkerArray()

        # Grasp approach arrow
        approach_arrow = Marker()
        approach_arrow.header.frame_id = grasp_pose.header.frame_id
        approach_arrow.header.stamp = grasp_pose.header.stamp
        approach_arrow.ns = "grasp_approach"
        approach_arrow.id = 0
        approach_arrow.type = Marker.ARROW
        approach_arrow.action = Marker.ADD

        # Set start and end points for arrow
        approach_arrow.points = []
        start_point = PointStamped()
        start_point.point.x = grasp_pose.pose.position.x
        start_point.point.y = grasp_pose.pose.position.y
        start_point.point.z = grasp_pose.pose.position.z
        approach_arrow.points.append(start_point.point)

        # End point is slightly offset in the approach direction
        end_point = PointStamped()
        # This is a simplified approach - real implementation would use gripper approach vector
        end_point.point.x = grasp_pose.pose.position.x + 0.1
        end_point.point.y = grasp_pose.pose.position.y
        end_point.point.z = grasp_pose.pose.position.z
        approach_arrow.points.append(end_point.point)

        approach_arrow.scale.x = 0.02  # Shaft diameter
        approach_arrow.scale.y = 0.04  # Head diameter
        approach_arrow.color.a = 1.0
        approach_arrow.color.r = 1.0
        approach_arrow.color.g = 0.0
        approach_arrow.color.b = 0.0

        marker_array.markers.append(approach_arrow)

        self.visualization_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    manip_pipeline = IsaacManipulationPipeline()

    try:
        rclpy.spin(manip_pipeline)
    except KeyboardInterrupt:
        manip_pipeline.get_logger().info('Shutting down Isaac Manipulation Pipeline')
    finally:
        manip_pipeline.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 8. Performance Optimization for Isaac ROS

### Optimized Isaac ROS Configuration:
```python
# performance_optimizer.py
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Int32, Float32
import numpy as np
import threading
import time

class IsaacROSOptimizer(Node):
    def __init__(self):
        super().__init__('isaac_ros_optimizer')

        # Performance metrics
        self.fps = Float32()
        self.gpu_utilization = Float32()
        self.memory_usage = Float32()

        # Publishers for performance metrics
        self.fps_pub = self.create_publisher(Float32, '/pipeline_fps', 10)
        self.gpu_util_pub = self.create_publisher(Float32, '/gpu_utilization', 10)
        self.mem_util_pub = self.create_publisher(Float32, '/memory_usage', 10)

        # Configuration parameters
        self.pipeline_config = {
            'image_processing_rate': 30,  # Hz
            'detection_confidence': 0.7,
            'tracking_threshold': 10,
            'max_objects': 50
        }

        # Frame timing
        self.frame_times = []
        self.max_frame_times = 100

        # Timers
        self.monitor_timer = self.create_timer(1.0, self.monitor_performance)

        # Threading for parallel processing
        self.processing_queue = []
        self.queue_lock = threading.Lock()
        self.processing_thread = threading.Thread(target=self.process_pipeline_parallel)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        self.get_logger().info('Isaac ROS Optimizer initialized')

    def monitor_performance(self):
        """Monitor pipeline performance"""
        if len(self.frame_times) > 1:
            avg_time = sum(self.frame_times) / len(self.frame_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0

            self.fps.data = fps
            self.fps_pub.publish(self.fps)

            # Simulate GPU and memory monitoring
            # In real Isaac ROS, this would interface with nvidia-ml-py
            self.gpu_utilization.data = np.random.uniform(40, 80)  # 40-80% utilization
            self.memory_usage.data = np.random.uniform(60, 85)    # 60-85% memory usage

            self.gpu_util_pub.publish(self.gpu_utilization)
            self.mem_util_pub.publish(self.memory_usage)

            self.get_logger().info(f'Pipeline FPS: {fps:.2f}, GPU Util: {self.gpu_utilization.data:.1f}%')

    def optimize_pipeline_config(self):
        """Adjust pipeline configuration based on performance"""
        # Monitor current performance
        current_fps = self.fps.data if hasattr(self, 'fps') else 0

        if current_fps < 15:  # Too slow, reduce quality
            self.pipeline_config['image_processing_rate'] = max(10, self.pipeline_config['image_processing_rate'] - 5)
            self.pipeline_config['detection_confidence'] = min(0.9, self.pipeline_config['detection_confidence'] + 0.05)
            self.get_logger().warn(f'Performance low, reducing to {self.pipeline_config["image_processing_rate"]}Hz')

        elif current_fps > 25:  # Good performance, can increase quality
            self.pipeline_config['image_processing_rate'] = min(60, self.pipeline_config['image_processing_rate'] + 2)
            self.pipeline_config['detection_confidence'] = max(0.5, self.pipeline_config['detection_confidence'] - 0.02)
            self.get_logger().info(f'Good performance, increasing to {self.pipeline_config["image_processing_rate"]}Hz')

    def process_pipeline_parallel(self):
        """Process pipeline in parallel thread"""
        while rclpy.ok():
            with self.queue_lock:
                if self.processing_queue:
                    item = self.processing_queue.pop(0)
                    # Process the item
                    self.process_pipeline_item(item)

            time.sleep(0.001)  # Small sleep to prevent busy waiting

    def process_pipeline_item(self, item):
        """Process individual pipeline item"""
        # This would process the actual pipeline item
        # For demonstration, we'll just measure timing
        start_time = time.time()

        # Simulate processing time based on configuration
        processing_delay = 1.0 / self.pipeline_config['image_processing_rate']
        time.sleep(min(processing_delay, 0.1))  # Cap at 100ms

        end_time = time.time()

        # Update frame timing
        frame_time = end_time - start_time
        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.max_frame_times:
            self.frame_times.pop(0)

        # Adjust configuration based on performance
        self.optimize_pipeline_config()

def main(args=None):
    rclpy.init(args=args)
    optimizer = IsaacROSOptimizer()

    try:
        rclpy.spin(optimizer)
    except KeyboardInterrupt:
        optimizer.get_logger().info('Shutting down Isaac ROS Optimizer')
    finally:
        optimizer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 9. Isaac ROS Integration with Isaac Sim

### Sim-to-Real Transfer Setup:
```python
# sim_to_real_transfer.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, JointState
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Bool, String
from cv_bridge import CvBridge
import numpy as np
import cv2
from collections import deque

class SimToRealTransfer(Node):
    def __init__(self):
        super().__init__('sim_to_real_transfer')

        # Initialize bridge
        self.bridge = CvBridge()

        # Domain randomization parameters
        self.domain_randomization = {
            'lighting_variation': 0.3,
            'texture_randomization': True,
            'occlusion_probability': 0.1,
            'motion_blur': True
        }

        # Sim-to-real adaptation
        self.sim_data_buffer = deque(maxlen=100)
        self.real_data_buffer = deque(maxlen=100)

        # Subscribers for both sim and real data
        self.sim_image_sub = self.create_subscription(
            Image, '/sim/camera/rgb/image_raw', self.sim_image_callback, 10
        )
        self.real_image_sub = self.create_subscription(
            Image, '/real/camera/rgb/image_raw', self.real_image_callback, 10
        )

        # Publishers for adapted data
        self.adapted_image_pub = self.create_publisher(Image, '/adapted/image', 10)
        self.transfer_status_pub = self.create_publisher(String, '/transfer_status', 10)

        # Parameters for domain adaptation
        self.adaptation_enabled = True
        self.color_correction = np.ones(3, dtype=np.float32)  # RGB gain factors
        self.exposure_compensation = 0.0

        self.get_logger().info('Sim-to-Real Transfer node initialized')

    def sim_image_callback(self, msg):
        """Process simulation image data"""
        if not self.adaptation_enabled:
            return

        # Convert to OpenCV
        cv_sim_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Apply domain randomization to simulation data
        randomized_image = self.apply_domain_randomization(cv_sim_image)

        # Store in buffer for comparison
        self.sim_data_buffer.append(randomized_image)

        # Adapt image for real-world deployment
        adapted_image = self.adapt_sim_to_real(randomized_image)

        # Publish adapted image
        adapted_msg = self.bridge.cv2_to_imgmsg(adapted_image, encoding='bgr8')
        adapted_msg.header = msg.header
        self.adapted_image_pub.publish(adapted_msg)

    def real_image_callback(self, msg):
        """Process real-world image data"""
        if not self.adaptation_enabled:
            return

        # Convert to OpenCV
        cv_real_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Store in buffer for comparison
        self.real_data_buffer.append(cv_real_image)

        # Update adaptation parameters based on real data
        self.update_adaptation_parameters(cv_real_image)

    def apply_domain_randomization(self, image):
        """Apply domain randomization to simulation image"""
        # Randomize lighting
        lighting_factor = 1.0 + np.random.uniform(
            -self.domain_randomization['lighting_variation'],
            self.domain_randomization['lighting_variation']
        )
        randomized = np.clip(image.astype(np.float32) * lighting_factor, 0, 255).astype(np.uint8)

        # Add random noise
        noise = np.random.normal(0, 5, randomized.shape).astype(np.float32)
        randomized = np.clip(randomized.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Add motion blur occasionally
        if self.domain_randomization['motion_blur'] and np.random.random() < 0.1:
            kernel_size = np.random.randint(2, 5)
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[int(kernel_size/2), :] = 1.0 / kernel_size
            randomized = cv2.filter2D(randomized, -1, kernel)

        return randomized

    def adapt_sim_to_real(self, sim_image):
        """Adapt simulation image to be more similar to real-world images"""
        # Apply color correction
        adapted = sim_image.astype(np.float32)
        adapted[:, :, 0] *= self.color_correction[0]  # Blue channel
        adapted[:, :, 1] *= self.color_correction[1]  # Green channel
        adapted[:, :, 2] *= self.color_correction[2]  # Red channel

        # Apply exposure compensation
        adapted = adapted * (1.0 + self.exposure_compensation)

        # Clip values
        adapted = np.clip(adapted, 0, 255).astype(np.uint8)

        return adapted

    def update_adaptation_parameters(self, real_image):
        """Update adaptation parameters based on real-world data"""
        if len(self.sim_data_buffer) == 0:
            return

        # Get average color from real image
        real_avg_color = np.mean(real_image, axis=(0, 1))

        # Get average color from sim image (before adaptation)
        sim_avg_color = np.mean(self.sim_data_buffer[-1], axis=(0, 1))

        # Update color correction factors
        # Only update slowly to avoid oscillation
        for i in range(3):
            if sim_avg_color[i] > 0:
                target_correction = real_avg_color[i] / sim_avg_color[i]
                self.color_correction[i] = 0.95 * self.color_correction[i] + 0.05 * target_correction

        # Update exposure compensation
        real_brightness = np.mean(cv2.cvtColor(real_image, cv2.COLOR_BGR2GRAY))
        sim_brightness = np.mean(cv2.cvtColor(self.sim_data_buffer[-1], cv2.COLOR_BGR2GRAY))

        if sim_brightness > 0:
            target_exposure = (real_brightness / sim_brightness) - 1.0
            self.exposure_compensation = 0.9 * self.exposure_compensation + 0.1 * target_exposure

        self.get_logger().info(f'Updated adaptation parameters - Color correction: {self.color_correction}, Exposure: {self.exposure_compensation:.3f}')

def main(args=None):
    rclpy.init(args=args)
    transfer_node = SimToRealTransfer()

    try:
        rclpy.spin(transfer_node)
    except KeyboardInterrupt:
        transfer_node.get_logger().info('Shutting down Sim-to-Real Transfer')
    finally:
        transfer_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 10. Troubleshooting Isaac ROS Issues

### Common Issues and Solutions:

1. **GPU Memory Issues**:
   ```bash
   # Check GPU memory usage
   nvidia-smi

   # Reduce model input resolution
   # Use TensorRT optimization
   # Implement memory pooling
   ```

2. **Performance Bottlenecks**:
   ```bash
   # Profile with Nsight Systems
   nsys profile --trace=cuda,nvtx ros2 run ...

   # Use Isaac ROS benchmarks
   ros2 launch isaac_ros_benchmark benchmark.launch.py
   ```

3. **CUDA Runtime Errors**:
   ```bash
   # Verify CUDA installation
   nvcc --version
   nvidia-smi

   # Check Isaac ROS CUDA compatibility
   # Ensure proper driver versions
   ```

4. **Integration Issues**:
   ```bash
   # Check topic connections
   ros2 topic list
   ros2 topic echo /topic_name

   # Verify message types
   ros2 interface show sensor_msgs/msg/Image
   ```

## Best Practices for Isaac ROS

### Performance Optimization:
- Use TensorRT for neural network optimization
- Implement efficient data pipelines with shared memory
- Use appropriate QoS settings for real-time performance
- Optimize image resolution based on task requirements

### Robustness Considerations:
- Implement fallback mechanisms for perception failures
- Use sensor fusion for redundancy
- Validate outputs before using in control systems
- Monitor system health and performance metrics

### Development Workflow:
- Test in simulation before deploying to hardware
- Use Isaac Sim for data generation and testing
- Implement modular pipeline components
- Use version control for model and parameter files

## Weekly Schedule Focus (Weeks 8-10)
During Weeks 8-10, we will focus on:
- Isaac ROS: Hardware-accelerated VSLAM (Visual SLAM) and navigation
- Nav2: Path planning for bipedal humanoid movement
- Advanced perception and training systems
- Integration of Isaac Sim and Isaac ROS for sim-to-real transfer

## Resources
- [Isaac ROS Documentation](https://nvidia-isaac-ros.github.io/)
- [Isaac ROS GitHub Repository](https://github.com/NVIDIA-ISAAC-ROS)
- [Isaac ROS Navigation](https://nvidia-isaac-ros.github.io/concepts/navigation/index.html)
- [Isaac ROS Perception](https://nvidia-isaac-ros.github.io/concepts/perception/index.html)
- [NVIDIA Jetson Optimization Guide](https://docs.nvidia.com/jetson/l4t/index.html)
