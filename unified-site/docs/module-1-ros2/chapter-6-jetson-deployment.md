# Chapter 6: Jetson Deployment for Humanoid Robots

## Overview
This chapter covers the deployment of ROS 2 packages to NVIDIA Jetson platforms, which serve as the "brain" for humanoid robots. Students will learn to optimize and deploy their ROS 2 applications on edge computing devices, ensuring efficient resource utilization and real-time performance.

## Learning Objectives
By the end of this chapter, students will be able to:
- Set up and configure NVIDIA Jetson platforms for ROS 2
- Optimize ROS 2 packages for resource-constrained environments
- Deploy and run humanoid robot applications on Jetson devices
- Monitor and troubleshoot deployed systems
- Implement efficient communication between development and deployment environments

## 1. Introduction to NVIDIA Jetson for Robotics

NVIDIA Jetson platforms are industry-standard edge AI computing devices designed for robotics and AI applications. For humanoid robots, Jetson devices serve as the "brain" that processes sensor data, runs AI algorithms, and controls robot behavior.

### Jetson Platform Options for Humanoid Robotics:
- **Jetson Orin Nano**: 8GB or 16GB versions, perfect for basic humanoid control
- **Jetson Orin NX**: More powerful option for complex AI processing
- **Jetson AGX Orin**: High-performance option for advanced humanoid capabilities

### Key Specifications for Humanoid Robotics:
- **GPU**: NVIDIA Ampere architecture for AI inference
- **CPU**: ARM-based multi-core processors
- **Memory**: 4GB-64GB LPDDR5 for handling sensor data and control
- **Connectivity**: Multiple interfaces for sensors and actuators

## 2. Setting Up Jetson for ROS 2 Development

### Initial Setup
```bash
# Flash Jetson with appropriate OS
# Use NVIDIA SDK Manager for initial flashing

# Update system packages
sudo apt update && sudo apt upgrade -y

# Install ROS 2 dependencies
sudo apt install software-properties-common
sudo add-apt-repository universe

# Install ROS 2 Iron (or appropriate version for Jetson)
sudo apt update && sudo apt install -y curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install -y ros-iron-desktop python3-rosdep
```

### Jetson-Specific Optimizations
```bash
# Set Jetson to maximum performance mode
sudo nvpmodel -m 0

# Set fan to maximum speed for cooling during intensive operations
sudo jetson_clocks

# Install Jetson.GPIO for hardware interface
pip3 install Jetson.GPIO
```

## 3. Cross-Compilation and Deployment Strategies

### Cross-Compilation Setup
```bash
# On development machine - set up cross-compilation environment
# Install cross-compilation tools
sudo apt install crossbuild-essential-arm64

# Create a Docker container for cross-compilation
docker run --rm -it --platform linux/arm64 ubuntu:22.04 bash

# Inside the container, install ROS 2 Iron for ARM64
# Follow ROS 2 installation instructions for ARM64 architecture
```

### Build and Transfer Method
```bash
# Build packages on development machine with ARM64 target
colcon build --cmake-force-ctest-parallel-level 4 --parallel-workers 4

# Transfer built packages to Jetson
scp -r install/ jetson@jetson-ip:/home/jetson/ros2_ws/
```

### Native Build on Jetson
```bash
# On Jetson - build packages natively (slower but reliable)
cd ~/ros2_ws/src
git clone https://github.com/your-organization/humanoid_control.git

cd ~/ros2_ws
source /opt/ros/iron/setup.bash
colcon build --packages-select humanoid_control
```

## 4. Optimizing ROS 2 Packages for Jetson

### Memory Optimization Example
```python
# optimized_controller.py
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray
import psutil  # For monitoring resource usage

class OptimizedController(Node):
    def __init__(self):
        super().__init__('optimized_controller')

        # Use efficient data structures
        self.joint_positions = np.zeros(20, dtype=np.float32)  # Fixed size, single precision
        self.joint_velocities = np.zeros(20, dtype=np.float32)
        self.control_commands = np.zeros(20, dtype=np.float32)

        # Subscribers with appropriate QoS for real-time performance
        from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE
        )

        self.joint_state_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, qos_profile
        )

        self.control_cmd_pub = self.create_publisher(
            Float32MultiArray, 'optimized_commands', qos_profile
        )

        # Optimized control loop at 200Hz
        self.control_timer = self.create_timer(0.005, self.optimized_control_loop)

        self.get_logger().info('Optimized Controller initialized for Jetson deployment')

    def joint_state_callback(self, msg):
        """Optimized callback to minimize memory allocation"""
        # Direct assignment to pre-allocated arrays
        for i, pos in enumerate(msg.position):
            if i < len(self.joint_positions):
                self.joint_positions[i] = pos

        for i, vel in enumerate(msg.velocity):
            if i < len(self.joint_velocities):
                self.joint_velocities[i] = vel

    def optimized_control_loop(self):
        """Memory-efficient control computation"""
        # Vectorized computation using numpy
        self.control_commands = self.compute_control_vectorized()

        # Publish with pre-allocated message
        cmd_msg = Float32MultiArray()
        cmd_msg.data = self.control_commands.tolist()
        self.control_cmd_pub.publish(cmd_msg)

        # Monitor resource usage
        if self.get_clock().now().nanoseconds % 1000000000 < 50000000:  # Log every second
            memory_percent = psutil.virtual_memory().percent
            cpu_percent = psutil.cpu_percent()
            if memory_percent > 80:
                self.get_logger().warn(f'High memory usage: {memory_percent}%')
            if cpu_percent > 80:
                self.get_logger().warn(f'High CPU usage: {cpu_percent}%')

    def compute_control_vectorized(self):
        """Vectorized control computation"""
        # Example: simple PD control in vectorized form
        desired_positions = np.zeros_like(self.joint_positions)  # Placeholder
        position_error = desired_positions - self.joint_positions
        velocity_feedback = self.joint_velocities

        control_output = 10.0 * position_error - 1.0 * velocity_feedback
        return np.clip(control_output, -50.0, 50.0)  # Limit commands

def main(args=None):
    rclpy.init(args=args)
    controller = OptimizedController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down optimized controller')
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 5. Hardware Interface Optimization

### Efficient Sensor Data Processing
```python
# jetson_sensor_interface.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu, Image
import numpy as np
from collections import deque
import threading

class JetsonSensorInterface(Node):
    def __init__(self):
        super().__init__('jetson_sensor_interface')

        # Use circular buffers for efficient data storage
        self.joint_buffer = deque(maxlen=10)
        self.imu_buffer = deque(maxlen=10)

        # Hardware-specific optimizations for Jetson
        from rclpy.qos import QoSProfile
        qos_profile = QoSProfile(depth=1)

        # Subscribe to sensor data
        self.joint_sub = self.create_subscription(
            JointState, 'hardware_joint_states', self.joint_callback, qos_profile
        )
        self.imu_sub = self.create_subscription(
            Imu, 'hardware_imu', self.imu_callback, qos_profile
        )

        # Multi-threaded processing for better performance
        self.processing_lock = threading.Lock()
        self.latest_joint_data = None
        self.latest_imu_data = None

        self.get_logger().info('Jetson Sensor Interface initialized')

    def joint_callback(self, msg):
        """Process joint state data efficiently"""
        with self.processing_lock:
            # Convert to numpy arrays for efficient processing
            self.latest_joint_data = {
                'position': np.array(msg.position, dtype=np.float32),
                'velocity': np.array(msg.velocity, dtype=np.float32),
                'effort': np.array(msg.effort, dtype=np.float32),
                'names': msg.name
            }

        # Add to buffer for history
        self.joint_buffer.append(self.latest_joint_data)

    def imu_callback(self, msg):
        """Process IMU data efficiently"""
        with self.processing_lock:
            self.latest_imu_data = {
                'orientation': np.array([
                    msg.orientation.x, msg.orientation.y,
                    msg.orientation.z, msg.orientation.w
                ], dtype=np.float32),
                'angular_velocity': np.array([
                    msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z
                ], dtype=np.float32),
                'linear_acceleration': np.array([
                    msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z
                ], dtype=np.float32)
            }

        self.imu_buffer.append(self.latest_imu_data)

    def get_latest_sensor_data(self):
        """Thread-safe access to latest sensor data"""
        with self.processing_lock:
            return self.latest_joint_data, self.latest_imu_data
```

## 6. Deployment Configuration

### Launch File for Jetson Deployment
```python
# launch/jetson_humanoid_launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')

    # Set environment variables for Jetson optimization
    set_jetson_env = SetEnvironmentVariable(
        name='CUDA_VISIBLE_DEVICES',
        value='0'
    )

    set_omp_env = SetEnvironmentVariable(
        name='OMP_NUM_THREADS',
        value='4'  # Match Jetson CPU cores
    )

    return LaunchDescription([
        set_jetson_env,
        set_omp_env,

        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation clock if true'
        ),

        # Optimized controller node
        Node(
            package='humanoid_control',
            executable='optimized_controller',
            name='jetson_controller',
            parameters=[
                {'control_frequency': 200},  # Higher frequency for Jetson
                {'use_gpu_acceleration': True},
                {'max_memory_usage': 0.8},  # Use up to 80% of memory
            ],
            remappings=[
                ('joint_states', 'hardware_joint_states'),
                ('control_commands', 'hardware_control_commands'),
            ],
            output='screen',
            # Resource limits for Jetson
            arguments=['--log-level', 'INFO']
        ),

        # Sensor interface node
        Node(
            package='humanoid_hardware',
            executable='jetson_sensor_interface',
            name='jetson_sensor_interface',
            parameters=[
                {'buffer_size': 10},
                {'processing_threads': 2},
            ],
            output='screen'
        )
    ])
```

## 7. Monitoring and Debugging on Jetson

### Resource Monitoring Node
```python
# jetson_monitor.py
import rclpy
from rclpy.node import Node
import psutil
import GPUtil
from std_msgs.msg import String
import json

class JetsonMonitor(Node):
    def __init__(self):
        super().__init__('jetson_monitor')

        self.monitor_publisher = self.create_publisher(
            String, 'jetson_status', 10
        )

        # Monitor at 1Hz
        self.monitor_timer = self.create_timer(1.0, self.monitor_system)

        self.get_logger().info('Jetson Monitor initialized')

    def monitor_system(self):
        """Monitor system resources"""
        status = {
            'timestamp': self.get_clock().now().nanoseconds,
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'temperature': self.get_jetson_temperature(),
        }

        # Get GPU info if available
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                status['gpu_load'] = gpu.load * 100
                status['gpu_memory_percent'] = (gpu.memoryUsed / gpu.memoryTotal) * 100
        except:
            pass  # GPU monitoring not available

        # Publish status
        status_msg = String()
        status_msg.data = json.dumps(status)
        self.monitor_publisher.publish(status_msg)

        # Log warnings for high resource usage
        if status['cpu_percent'] > 90:
            self.get_logger().warn(f'High CPU usage: {status["cpu_percent"]}%')
        if status['memory_percent'] > 90:
            self.get_logger().warn(f'High memory usage: {status["memory_percent"]}%')

    def get_jetson_temperature(self):
        """Get Jetson temperature (may vary by model)"""
        try:
            # Different Jetson models have different temperature sensors
            temp_files = [
                '/sys/devices/virtual/thermal/thermal_zone0/temp',
                '/sys/devices/virtual/thermal/thermal_zone1/temp',
            ]

            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    with open(temp_file, 'r') as f:
                        temp = f.read().strip()
                        return float(temp) / 1000.0  # Convert from millidegrees to degrees
        except:
            pass
        return 0.0

def main(args=None):
    rclpy.init(args=args)
    monitor = JetsonMonitor()

    try:
        rclpy.spin(monitor)
    except KeyboardInterrupt:
        monitor.get_logger().info('Shutting down Jetson monitor')
    finally:
        monitor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 8. Performance Optimization Techniques

### Efficient Message Handling
```python
# efficient_message_handler.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import numpy as np

class EfficientMessageHandler(Node):
    def __init__(self):
        super().__init__('efficient_message_handler')

        # Pre-allocate message objects to reduce allocation overhead
        self.preallocated_joint_cmd = JointState()
        self.preallocated_header = Header()

        # Initialize with expected sizes
        self.preallocated_joint_cmd.name = ['joint_' + str(i) for i in range(20)]
        self.preallocated_joint_cmd.position = [0.0] * 20
        self.preallocated_joint_cmd.velocity = [0.0] * 20
        self.preallocated_joint_cmd.effort = [0.0] * 20

        # Subscriber
        self.joint_state_sub = self.create_subscription(
            JointState, 'joint_states', self.efficient_callback, 10
        )

        self.cmd_pub = self.create_publisher(JointState, 'joint_commands', 10)

        self.get_logger().info('Efficient Message Handler initialized')

    def efficient_callback(self, msg):
        """Process message efficiently with minimal allocation"""
        # Update pre-allocated message instead of creating new ones
        self.preallocated_header.stamp = self.get_clock().now().to_msg()
        self.preallocated_header.frame_id = 'base_link'

        # Copy data efficiently
        self.preallocated_joint_cmd.header = self.preallocated_header
        self.preallocated_joint_cmd.position = self.process_joint_data(msg.position)

        # Publish pre-allocated message
        self.cmd_pub.publish(self.preallocated_joint_cmd)

    def process_joint_data(self, joint_positions):
        """Efficient joint data processing"""
        # Convert to numpy for vectorized operations
        pos_array = np.array(joint_positions, dtype=np.float32)

        # Apply processing (example: simple filtering)
        processed_pos = pos_array * 0.99  # Dampening example

        return processed_pos.tolist()
```

## 9. Troubleshooting Common Jetson Deployment Issues

### Common Issues and Solutions:

1. **Memory Constraints**:
   - Use single precision (float32) instead of double precision (float64)
   - Implement circular buffers instead of growing lists
   - Use memory-mapped files for large datasets

2. **Performance Issues**:
   - Profile code to identify bottlenecks
   - Use numpy for vectorized operations
   - Implement multi-threading for I/O operations

3. **Hardware Interface Problems**:
   - Check GPIO permissions
   - Verify sensor connections
   - Use appropriate baud rates for serial communication

4. **Real-time Performance**:
   - Use real-time kernel if needed
   - Minimize dynamic memory allocation
   - Prioritize critical control loops

## Weekly Schedule Focus (Weeks 3-5)
During Weeks 3-5, we will focus on:
- Setting up Jetson platforms for ROS 2 development
- Optimizing packages for edge deployment
- Deploying and testing on physical hardware
- Performance monitoring and tuning

## Resources
- [NVIDIA Jetson Documentation](https://developer.nvidia.com/embedded/jetson-developer-kits)
- [ROS 2 on Jetson](https://docs.ros.org/en/rolling/Installation/Nvidia-Jetson.html)
- [Jetson Hardware Acceleration](https://github.com/NVIDIA-AI-IOT/jetson-gpio)
- [ROS 2 Performance Optimization](https://docs.ros.org/en/rolling/How-To-Guides/Performance-optimization.html)
