# Chapter 4: Sensor Simulation

## Overview
This chapter covers the simulation of various sensors used in humanoid robots, including LiDAR, depth cameras, IMUs, and force/torque sensors. Students will learn to configure realistic sensor models in Gazebo that accurately represent their real-world counterparts, enabling effective development and testing of perception and control algorithms.

## Learning Objectives
By the end of this chapter, students will be able to:
- Configure and simulate LiDAR sensors with realistic noise and performance characteristics
- Set up depth camera simulation for 3D perception tasks
- Model IMU sensors for balance and orientation estimation
- Simulate force/torque sensors for contact detection and manipulation
- Validate sensor simulation against real-world performance
- Optimize sensor configurations for computational efficiency

## 1. Introduction to Sensor Simulation

Sensor simulation is critical for creating realistic digital twins of humanoid robots. In simulation, virtual sensors must accurately model:
- **Sensor characteristics**: Field of view, resolution, range, accuracy
- **Noise models**: Realistic sensor noise and uncertainty
- **Environmental effects**: Lighting, weather, and surface properties
- **Computational performance**: Efficient simulation without sacrificing realism

### Sensor Categories for Humanoid Robots:
- **Vision sensors**: RGB cameras, depth cameras, stereo cameras
- **Range sensors**: LiDAR, sonar, infrared range sensors
- **Inertial sensors**: IMUs, gyroscopes, accelerometers
- **Force sensors**: Force/torque sensors, tactile sensors
- **Proprioceptive sensors**: Joint encoders, motor current sensors

## 2. LiDAR Sensor Simulation

### Basic LiDAR Configuration
```xml
<!-- LiDAR sensor in URDF -->
<link name="lidar_link">
  <inertial>
    <mass value="0.5"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
  </inertial>
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <cylinder radius="0.05" length="0.05"/>
    </geometry>
  </visual>
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <cylinder radius="0.05" length="0.05"/>
    </geometry>
  </collision>
</link>

<joint name="base_to_lidar" type="fixed">
  <parent link="base_link"/>
  <child link="lidar_link"/>
  <origin xyz="0 0 0.3" rpy="0 0 0"/>
</joint>

<!-- Gazebo LiDAR sensor -->
<gazebo reference="lidar_link">
  <sensor name="laser_scanner" type="ray">
    <always_on>true</always_on>
    <update_rate>10</update_rate>
    <ray>
      <scan>
        <horizontal>
          <samples>720</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle>  <!-- -π radians = -180 degrees -->
          <max_angle>3.14159</max_angle>   <!-- π radians = 180 degrees -->
        </horizontal>
      </scan>
      <range>
        <min>0.1</min>
        <max>30.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="laser_controller" filename="libgazebo_ros_ray_sensor.so">
      <ros>
        <namespace>/laser</namespace>
        <remapping>~/out:=scan</remapping>
      </ros>
      <output_type>sensor_msgs/LaserScan</output_type>
    </plugin>
  </sensor>
</gazebo>
```

### Advanced LiDAR with Multiple Layers (3D LiDAR)
```xml
<!-- 3D LiDAR sensor configuration -->
<gazebo reference="lidar_3d_link">
  <sensor name="velodyne_sensor" type="ray">
    <always_on>true</always_on>
    <update_rate>10</update_rate>
    <ray>
      <scan>
        <horizontal>
          <samples>800</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
        <vertical>
          <samples>32</samples>
          <resolution>1</resolution>
          <min_angle>-0.5236</min_angle>  <!-- -30 degrees -->
          <max_angle>0.2618</max_angle>    <!-- 15 degrees -->
        </vertical>
      </scan>
      <range>
        <min>0.1</min>
        <max>100.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="velodyne_controller" filename="libgazebo_ros_velodyne_gpu.so">
      <ros>
        <namespace>/velodyne</namespace>
        <remapping>~/out:=cloud</remapping>
      </ros>
      <output_type>sensor_msgs/PointCloud2</output_type>
      <min_range>0.9</min_range>
      <max_range>100.0</max_range>
      <gaussian_noise>0.008</gaussian_noise>
    </plugin>
  </sensor>
</gazebo>
```

## 3. Camera and Depth Sensor Simulation

### RGB Camera Configuration
```xml
<!-- RGB camera sensor -->
<link name="camera_link">
  <inertial>
    <mass value="0.1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
  </inertial>
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="0.05 0.05 0.05"/>
    </geometry>
  </visual>
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="0.05 0.05 0.05"/>
    </geometry>
  </collision>
</link>

<joint name="head_to_camera" type="fixed">
  <parent link="head"/>
  <child link="camera_link"/>
  <origin xyz="0.05 0 0.05" rpy="0 0 0"/>
</joint>

<!-- Gazebo RGB camera -->
<gazebo reference="camera_link">
  <sensor name="rgb_camera" type="camera">
    <always_on>true</always_on>
    <update_rate>30.0</update_rate>
    <camera name="head_camera">
      <horizontal_fov>1.3962634</horizontal_fov> <!-- 80 degrees -->
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>100</far>
      </clip>
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.007</stddev>
      </noise>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>camera_link</frame_name>
      <topic_name>camera/image_raw</topic_name>
      <hack_baseline>0.07</hack_baseline>
    </plugin>
  </sensor>
</gazebo>
```

### Depth Camera Configuration
```xml
<!-- Depth camera sensor -->
<gazebo reference="camera_link">
  <sensor name="depth_camera" type="depth">
    <always_on>true</always_on>
    <update_rate>30.0</update_rate>
    <camera name="depth_head_camera">
      <horizontal_fov>1.3962634</horizontal_fov> <!-- 80 degrees -->
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>10</far>
      </clip>
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.05</stddev>
      </noise>
    </camera>
    <plugin name="depth_camera_controller" filename="libgazebo_ros_openni_kinect.so">
      <baseline>0.2</baseline>
      <always_on>true</always_on>
      <update_rate>30.0</update_rate>
      <camera_name>head_depth_camera</camera_name>
      <image_topic_name>rgb/image_raw</image_topic_name>
      <depth_image_topic_name>depth/image_raw</depth_image_topic_name>
      <point_cloud_topic_name>depth/points</point_cloud_topic_name>
      <camera_info_topic_name>rgb/camera_info</camera_info_topic_name>
      <depth_image_camera_info_topic_name>depth/camera_info</depth_image_camera_info_topic_name>
      <point_cloud_cutoff>0.1</point_cloud_cutoff>
      <point_cloud_cutoff_max>10.0</point_cloud_cutoff_max>
      <frame_name>camera_depth_optical_frame</frame_name>
      <distortion_k1>0.0</distortion_k1>
      <distortion_k2>0.0</distortion_k2>
      <distortion_k3>0.0</distortion_k3>
      <distortion_t1>0.0</distortion_t1>
      <distortion_t2>0.0</distortion_t2>
      <Cx_prime>0.0</Cx_prime>
      <Cx>320.5</Cx>
      <Cy>240.5</Cy>
      <focal_length>320.0</focal_length>
    </plugin>
  </sensor>
</gazebo>
```

### Stereo Camera Configuration
```xml
<!-- Stereo camera setup -->
<gazebo reference="camera_link">
  <sensor name="stereo_camera" type="multicamera">
    <always_on>true</always_on>
    <update_rate>30.0</update_rate>
    <camera name="left">
      <horizontal_fov>1.3962634</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>300</far>
      </clip>
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.007</stddev>
      </noise>
    </camera>
    <camera name="right">
      <horizontal_fov>1.3962634</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>300</far>
      </clip>
      <pose>0.1 0 0 0 0 0</pose> <!-- 10cm baseline -->
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.007</stddev>
      </noise>
    </camera>
    <plugin name="stereo_camera_controller" filename="libgazebo_ros_multicamera.so">
      <baseline>0.1</baseline>
      <always_on>true</always_on>
      <update_rate>30.0</update_rate>
      <camera_name>stereo_camera</camera_name>
      <image_topic_name>image_raw</image_topic_name>
      <camera_info_topic_name>camera_info</camera_info_topic_name>
      <left_frame_name>stereo_left_frame</left_frame_name>
      <right_frame_name>stereo_right_frame</right_frame_name>
      <hack_baseline>0.07</hack_baseline>
    </plugin>
  </sensor>
</gazebo>
```

## 4. IMU Sensor Simulation

### Basic IMU Configuration
```xml
<!-- IMU sensor configuration -->
<gazebo reference="torso">
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
            <bias_mean>0.0000075</bias_mean>
            <bias_stddev>0.0000008</bias_stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
            <bias_mean>0.0000075</bias_mean>
            <bias_stddev>0.0000008</bias_stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
            <bias_mean>0.0000075</bias_mean>
            <bias_stddev>0.0000008</bias_stddev>
          </noise>
        </z>
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
            <bias_mean>0.1</bias_mean>
            <bias_stddev>0.001</bias_stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
            <bias_mean>0.1</bias_mean>
            <bias_stddev>0.001</bias_stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
            <bias_mean>0.1</bias_mean>
            <bias_stddev>0.001</bias_stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
    <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
      <ros>
        <namespace>/imu</namespace>
        <remapping>~/out:=data</remapping>
      </ros>
      <frame_name>torso</frame_name>
      <initial_orientation_as_reference>false</initial_orientation_as_reference>
    </plugin>
  </sensor>
</gazebo>
```

### Multiple IMUs for Humanoid Balance
```xml
<!-- IMU in head for orientation -->
<gazebo reference="head">
  <sensor name="head_imu" type="imu">
    <always_on>true</always_on>
    <update_rate>200</update_rate>
    <imu>
      <angular_velocity>
        <x><noise type="gaussian"><stddev>0.001</stddev></noise></x>
        <y><noise type="gaussian"><stddev>0.001</stddev></noise></y>
        <z><noise type="gaussian"><stddev>0.001</stddev></noise></z>
      </angular_velocity>
      <linear_acceleration>
        <x><noise type="gaussian"><stddev>0.017</stddev></noise></x>
        <y><noise type="gaussian"><stddev>0.017</stddev></noise></y>
        <z><noise type="gaussian"><stddev>0.017</stddev></noise></z>
      </linear_acceleration>
    </imu>
    <plugin name="head_imu_plugin" filename="libgazebo_ros_imu.so">
      <ros><namespace>/head_imu</namespace></ros>
      <frame_name>head</frame_name>
    </plugin>
  </sensor>
</gazebo>

<!-- IMU in torso for balance -->
<gazebo reference="torso">
  <sensor name="torso_imu" type="imu">
    <always_on>true</always_on>
    <update_rate>200</update_rate>
    <imu>
      <angular_velocity>
        <x><noise type="gaussian"><stddev>0.001</stddev></noise></x>
        <y><noise type="gaussian"><stddev>0.001</stddev></noise></y>
        <z><noise type="gaussian"><stddev>0.001</stddev></noise></z>
      </angular_velocity>
      <linear_acceleration>
        <x><noise type="gaussian"><stddev>0.017</stddev></noise></x>
        <y><noise type="gaussian"><stddev>0.017</stddev></noise></y>
        <z><noise type="gaussian"><stddev>0.017</stddev></noise></z>
      </linear_acceleration>
    </imu>
    <plugin name="torso_imu_plugin" filename="libgazebo_ros_imu.so">
      <ros><namespace>/torso_imu</namespace></ros>
      <frame_name>torso</frame_name>
    </plugin>
  </sensor>
</gazebo>
```

## 5. Force/Torque Sensor Simulation

### Joint Force/Torque Sensors
```xml
<!-- Adding force/torque sensors to joints -->
<gazebo>
  <joint name="left_ankle">
    <sensor name="left_ankle_ft" type="force_torque">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <force_torque>
        <frame>child</frame>
        <measure_direction>child_to_parent</measure_direction>
      </force_torque>
    </sensor>
  </joint>
</gazebo>

<!-- Force/Torque sensor plugin -->
<gazebo>
  <plugin name="left_ankle_ft_plugin" filename="libgazebo_ros_ft_sensor.so">
    <ros>
      <namespace>/left_leg</namespace>
      <remapping>~/wrench:=left_ankle_wrench</remapping>
    </ros>
    <joint_name>left_ankle</joint_name>
    <frame_name>left_foot</frame_name>
    <gaussian_noise>0.01</gaussian_noise>
  </plugin>
</gazebo>
```

### Foot Contact Sensors
```xml
<!-- Contact sensors for feet -->
<gazebo reference="left_foot">
  <sensor name="left_foot_contact" type="contact">
    <always_on>true</always_on>
    <update_rate>1000</update_rate>
    <contact>
      <collision>left_foot_collision</collision>
    </contact>
    <plugin name="left_foot_contact_plugin" filename="libgazebo_ros_bumper.so">
      <ros>
        <namespace>/left_foot</namespace>
        <remapping>~/out:=contact</remapping>
      </ros>
      <frame_name>left_foot</frame_name>
    </plugin>
  </sensor>
</gazebo>

<gazebo reference="right_foot">
  <sensor name="right_foot_contact" type="contact">
    <always_on>true</always_on>
    <update_rate>1000</update_rate>
    <contact>
      <collision>right_foot_collision</collision>
    </contact>
    <plugin name="right_foot_contact_plugin" filename="libgazebo_ros_bumper.so">
      <ros>
        <namespace>/right_foot</namespace>
        <remapping>~/out:=contact</remapping>
      </ros>
      <frame_name>right_foot</frame_name>
    </plugin>
  </sensor>
</gazebo>
```

## 6. Sensor Fusion and Processing

### Sensor Data Processing Node
```python
# sensor_fusion.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image, Imu, PointCloud2, JointState
from geometry_msgs.msg import Vector3, Twist
from std_msgs.msg import Float32MultiArray
import numpy as np
import cv2
from cv_bridge import CvBridge

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion')

        # Initialize data storage
        self.laser_data = None
        self.imu_data = None
        self.camera_data = None
        self.joint_states = None

        # Subscribers for all sensors
        self.laser_sub = self.create_subscription(
            LaserScan, '/laser/scan', self.laser_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )
        self.camera_sub = self.create_subscription(
            Image, '/camera/image_raw', self.camera_callback, 10
        )
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10
        )

        # Publisher for fused data
        self.fused_data_pub = self.create_publisher(
            Float32MultiArray, '/sensor_fusion/output', 10
        )

        # Timer for fusion processing
        self.fusion_timer = self.create_timer(0.033, self.fusion_callback)  # 30Hz

        self.bridge = CvBridge()
        self.get_logger().info('Sensor Fusion Node initialized')

    def laser_callback(self, msg):
        """Process laser scan data"""
        self.laser_data = {
            'ranges': np.array(msg.ranges),
            'intensities': np.array(msg.intensities),
            'angle_min': msg.angle_min,
            'angle_max': msg.angle_max,
            'angle_increment': msg.angle_increment
        }

    def imu_callback(self, msg):
        """Process IMU data"""
        self.imu_data = {
            'orientation': [
                msg.orientation.x, msg.orientation.y,
                msg.orientation.z, msg.orientation.w
            ],
            'angular_velocity': [
                msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z
            ],
            'linear_acceleration': [
                msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z
            ]
        }

    def camera_callback(self, msg):
        """Process camera data"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # Process image for features, obstacles, etc.
            processed_data = self.process_camera_image(cv_image)
            self.camera_data = processed_data
        except Exception as e:
            self.get_logger().error(f'Error processing camera data: {e}')

    def joint_callback(self, msg):
        """Process joint state data"""
        self.joint_states = {
            'position': np.array(msg.position),
            'velocity': np.array(msg.velocity),
            'effort': np.array(msg.effort),
            'name': msg.name
        }

    def process_camera_image(self, image):
        """Process camera image for relevant information"""
        # Example: Simple obstacle detection using depth
        height, width = image.shape[:2]

        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Simple processing - in reality this would involve more complex computer vision
        processed = {
            'image_shape': (height, width),
            'mean_intensity': np.mean(gray),
            'edges_detected': cv2.Canny(gray, 50, 150).sum()
        }

        return processed

    def fusion_callback(self):
        """Main fusion algorithm"""
        if not all([self.laser_data, self.imu_data, self.joint_states]):
            return  # Wait for all sensors to have data

        # Perform sensor fusion
        fused_output = self.perform_fusion()

        # Publish fused data
        msg = Float32MultiArray()
        msg.data = fused_output
        self.fused_data_pub.publish(msg)

    def perform_fusion(self):
        """Perform sensor fusion calculations"""
        # Example fusion algorithm combining sensor data
        # This is a simplified example - real fusion would be much more complex

        # Get robot orientation from IMU
        imu_orientation = self.imu_data['orientation']

        # Get laser data for obstacle detection
        laser_ranges = self.laser_data['ranges']
        min_distance = np.min(laser_ranges[np.isfinite(laser_ranges)]) if np.any(np.isfinite(laser_ranges)) else float('inf')

        # Get joint positions for balance state
        if self.joint_states and 'left_ankle' in self.joint_states['name']:
            ankle_idx = self.joint_states['name'].index('left_ankle')
            ankle_position = self.joint_states['position'][ankle_idx]
        else:
            ankle_position = 0.0

        # Combine into a fused state vector
        fused_vector = [
            min_distance,  # Closest obstacle distance
            imu_orientation[0],  # Orientation components
            imu_orientation[1],
            imu_orientation[2],
            imu_orientation[3],
            ankle_position,  # Joint state for balance
        ]

        return fused_vector

def main(args=None):
    rclpy.init(args=args)
    fusion_node = SensorFusionNode()

    try:
        rclpy.spin(fusion_node)
    except KeyboardInterrupt:
        fusion_node.get_logger().info('Shutting down sensor fusion node')
    finally:
        fusion_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 7. Sensor Validation and Calibration

### Sensor Validation Tools
```python
# sensor_validation.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu, Image
import numpy as np
import statistics

class SensorValidator(Node):
    def __init__(self):
        super().__init__('sensor_validator')

        # Statistics for sensor validation
        self.scan_stats = {'ranges': [], 'intensities': []}
        self.imu_stats = {'acceleration': [], 'gyro': []}

        # Subscribers for validation
        self.scan_sub = self.create_subscription(
            LaserScan, '/laser/scan', self.validate_scan, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.validate_imu, 10
        )

        # Timer for validation reports
        self.validation_timer = self.create_timer(5.0, self.report_validation)

        self.get_logger().info('Sensor Validator initialized')

    def validate_scan(self, msg):
        """Validate laser scan data"""
        # Check for expected range values
        valid_ranges = [r for r in msg.ranges if np.isfinite(r) and 0.1 <= r <= 30.0]

        if len(valid_ranges) < len(msg.ranges) * 0.5:  # Less than 50% valid
            self.get_logger().warn('Laser scan has too many invalid ranges')

        # Store statistics
        self.scan_stats['ranges'].extend(valid_ranges)
        if len(self.scan_stats['ranges']) > 1000:  # Keep last 1000 readings
            self.scan_stats['ranges'] = self.scan_stats['ranges'][-1000:]

    def validate_imu(self, msg):
        """Validate IMU data"""
        # Check for reasonable acceleration values (should be around 9.8 for Z-axis when stable)
        acc_mag = np.sqrt(
            msg.linear_acceleration.x**2 +
            msg.linear_acceleration.y**2 +
            msg.linear_acceleration.z**2
        )

        if acc_mag > 20.0:  # Too high, might indicate sensor issue
            self.get_logger().warn(f'High acceleration magnitude detected: {acc_mag}')

        # Store statistics
        self.imu_stats['acceleration'].append([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])

        if len(self.imu_stats['acceleration']) > 1000:
            self.imu_stats['acceleration'] = self.imu_stats['acceleration'][-1000:]

    def report_validation(self):
        """Report sensor validation statistics"""
        if self.scan_stats['ranges']:
            range_mean = statistics.mean(self.scan_stats['ranges'])
            range_stdev = statistics.stdev(self.scan_stats['ranges']) if len(self.scan_stats['ranges']) > 1 else 0
            self.get_logger().info(f'Laser scan - Mean: {range_mean:.2f}, StdDev: {range_stdev:.2f}')

        if self.imu_stats['acceleration']:
            acc_array = np.array(self.imu_stats['acceleration'])
            acc_mean = np.mean(acc_array, axis=0)
            acc_stdev = np.std(acc_array, axis=0)
            self.get_logger().info(f'IMU acceleration - Mean: {acc_mean}, StdDev: {acc_stdev}')

def main(args=None):
    rclpy.init(args=args)
    validator = SensorValidator()

    try:
        rclpy.spin(validator)
    except KeyboardInterrupt:
        validator.get_logger().info('Shutting down sensor validator')
    finally:
        validator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 8. Performance Optimization for Sensor Simulation

### Efficient Sensor Configuration
```xml
<!-- Optimized sensor configuration for performance -->
<!-- Reduce update rates for sensors that don't need high frequency -->
<gazebo reference="camera_link">
  <sensor name="low_freq_camera" type="camera">
    <always_on>true</always_on>
    <update_rate>10.0</update_rate>  <!-- Lower update rate for performance -->
    <!-- ... rest of camera config ... -->
  </sensor>
</gazebo>

<!-- Use sensor noise models efficiently -->
<gazebo reference="imu_sensor">
  <sensor name="optimized_imu" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>  <!-- Balance accuracy with performance -->
    <imu>
      <!-- Use realistic but not overly complex noise models -->
      <angular_velocity>
        <x><noise type="gaussian"><stddev>0.001</stddev></noise></x>
      </angular_velocity>
      <linear_acceleration>
        <z><noise type="gaussian"><stddev>0.017</stddev></noise></z>
      </linear_acceleration>
    </imu>
  </sensor>
</gazebo>
```

### Dynamic Sensor Management
```python
# dynamic_sensor_manager.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import LaserScan, Image

class DynamicSensorManager(Node):
    def __init__(self):
        super().__init__('dynamic_sensor_manager')

        # Sensor activation flags
        self.laser_active = True
        self.camera_active = True

        # Publishers for sensor activation
        self.laser_enable_pub = self.create_publisher(Bool, '/laser/enable', 10)
        self.camera_enable_pub = self.create_publisher(Bool, '/camera/enable', 10)

        # Timer for dynamic management
        self.management_timer = self.create_timer(1.0, self.manage_sensors)

        self.get_logger().info('Dynamic Sensor Manager initialized')

    def manage_sensors(self):
        """Dynamically manage sensor activation based on needs"""
        # Example: Disable camera when not needed for power/performance
        # This would be based on robot state, mission requirements, etc.

        # For now, just maintain current state
        laser_msg = Bool()
        laser_msg.data = self.laser_active
        self.laser_enable_pub.publish(laser_msg)

        camera_msg = Bool()
        camera_msg.data = self.camera_active
        self.camera_enable_pub.publish(camera_msg)

def main(args=None):
    rclpy.init(args=args)
    manager = DynamicSensorManager()

    try:
        rclpy.spin(manager)
    except KeyboardInterrupt:
        manager.get_logger().info('Shutting down dynamic sensor manager')
    finally:
        manager.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 9. Troubleshooting Sensor Simulation Issues

### Common Sensor Issues and Solutions:

1. **Sensor data not publishing**:
   - Check that sensor plugin names and filenames are correct
   - Verify that ROS namespaces and topic names are properly configured
   - Ensure sensor links are properly connected in URDF

2. **High CPU/GPU usage from sensors**:
   - Reduce sensor update rates
   - Lower image resolution for cameras
   - Use fewer laser scan samples
   - Disable unused sensors

3. **Unrealistic sensor noise**:
   - Adjust noise parameters to match real sensor specifications
   - Verify that noise types and values are appropriate
   - Check that noise is being applied correctly in the simulation

4. **Sensor data timing issues**:
   - Synchronize sensor update rates with control loops
   - Check for proper time synchronization in simulation
   - Verify that sensor data timestamps are accurate

### Sensor Diagnostics Commands
```bash
# Check sensor topics
ros2 topic list | grep -E "(scan|image|imu|camera|depth)"

# Monitor sensor data rates
ros2 topic hz /laser/scan
ros2 topic hz /camera/image_raw
ros2 topic hz /imu/data

# Visualize sensor data
rviz2  # Then add sensor displays

# Check sensor plugins
gz topic -l  # List all Gazebo topics
```

## 10. Best Practices for Sensor Simulation

### Accuracy Considerations:
- Use realistic noise models based on actual sensor specifications
- Match sensor parameters (FOV, range, resolution) to real hardware
- Include environmental effects where appropriate
- Validate simulated sensors against real sensor data

### Performance Optimization:
- Use appropriate update rates for each sensor type
- Reduce resolution when high detail is not needed
- Disable sensors when not in use
- Use efficient sensor fusion algorithms

### Integration Tips:
- Ensure consistent coordinate frames across all sensors
- Synchronize sensor timestamps when needed for fusion
- Use standard ROS message types for compatibility
- Document sensor configurations and parameters

## Weekly Schedule Focus (Weeks 6-7)
During Weeks 6-7, we will focus on:
- Simulating sensors: LiDAR, Depth Cameras, and IMUs
- Physics simulation and sensor simulation
- Integration of sensors in Gazebo simulation
- Human-robot interaction in simulation environments

## Resources
- [Gazebo Sensor Documentation](http://gazebosim.org/tutorials?tut=ros_gzplugins#Sensor-plugins)
- [ROS 2 Sensor Integration](https://github.com/ros-simulation/gazebo_ros_pkgs/wiki)
- [Sensor Noise Modeling](http://gazebosim.org/tutorials?tut=ros_depth_camera&cat=sensors)
- [Humanoid Sensor Simulation Best Practices](https://humanoid-walk.readthedocs.io/)
