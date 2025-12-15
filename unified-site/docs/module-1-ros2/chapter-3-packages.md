# Chapter 3: Building ROS 2 Packages with Python

## Overview
This chapter covers the creation and management of ROS 2 packages using Python. Students will learn to build modular, reusable components for humanoid robot applications, following ROS 2 best practices and conventions.

## Learning Objectives
By the end of this chapter, students will be able to:
- Create ROS 2 packages using colcon build system
- Structure Python code following ROS 2 conventions
- Implement reusable nodes, libraries, and interfaces
- Manage package dependencies and configurations
- Build and test packages for humanoid robot applications

## 1. ROS 2 Package Structure

A typical ROS 2 package follows a standardized structure:

```
humanoid_control/
├── CMakeLists.txt          # Build configuration for C++
├── package.xml             # Package metadata
├── setup.py                # Python package configuration
├── setup.cfg               # Installation configuration
├── resource/               # Resource files
├── test/                   # Test files
├── launch/                 # Launch files
├── config/                 # Configuration files
└── humanoid_control/       # Python source code
    ├── __init__.py
    ├── controller.py
    └── utils.py
```

### Key Files Explained:
- **package.xml**: Contains package metadata, dependencies, and maintainers
- **setup.py**: Python package setup configuration
- **setup.cfg**: Installation instructions for Python packages
- **CMakeLists.txt**: Build configuration (required even for Python-only packages)

## 2. Creating a ROS 2 Package

### Using the ROS 2 Command Line Tool
```bash
# Create a new Python package
ros2 pkg create --build-type ament_python humanoid_control --dependencies rclpy std_msgs sensor_msgs geometry_msgs

# This creates the basic package structure
```

### Package.xml Configuration
```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>humanoid_control</name>
  <version>0.0.1</version>
  <description>Humanoid robot control package</description>
  <maintainer email="student@university.edu">Student Name</maintainer>
  <license>MIT</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>geometry_msgs</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

## 3. Python Package Configuration

### setup.py for Python Packages
```python
from setuptools import setup
from glob import glob
import os

package_name = 'humanoid_control'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # Include config files
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Student Name',
    maintainer_email='student@university.edu',
    description='Humanoid robot control package',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'humanoid_controller = humanoid_control.controller:main',
            'balance_node = humanoid_control.balance_node:main',
            'vision_processor = humanoid_control.vision_processor:main',
        ],
    },
)
```

## 4. Implementing Reusable Components

### Example: Humanoid Controller Package
```python
# humanoid_control/controller.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import numpy as np

class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')

        # Publishers
        self.joint_cmd_publisher = self.create_publisher(JointState, 'joint_commands', 10)
        self.status_publisher = self.create_publisher(String, 'robot_status', 10)

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10
        )
        self.cmd_vel_sub = self.create_subscription(
            Twist, 'cmd_vel', self.cmd_vel_callback, 10
        )

        # Timer for control loop
        self.control_timer = self.create_timer(0.05, self.control_loop)  # 20 Hz

        # Robot state
        self.current_joint_states = JointState()
        self.desired_velocity = Twist()

        self.get_logger().info('Humanoid Controller initialized')

    def joint_state_callback(self, msg):
        self.current_joint_states = msg

    def cmd_vel_callback(self, msg):
        self.desired_velocity = msg

    def control_loop(self):
        # Implement control logic here
        joint_cmd = self.compute_joint_commands()
        self.joint_cmd_publisher.publish(joint_cmd)

    def compute_joint_commands(self):
        # Placeholder for control algorithm
        cmd = JointState()
        cmd.name = self.current_joint_states.name
        cmd.position = [0.0] * len(self.current_joint_states.name)  # Placeholder
        return cmd

def main(args=None):
    rclpy.init(args=args)
    controller = HumanoidController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 5. Launch Files and Parameter Management

### Example Launch File
```python
# launch/humanoid_control_launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation clock if true'
        ),

        # Humanoid controller node
        Node(
            package='humanoid_control',
            executable='humanoid_controller',
            name='humanoid_controller',
            parameters=[
                {'kp': 1.0},
                {'ki': 0.1},
                {'kd': 0.05},
                {'max_velocity': 1.0},
            ],
            remappings=[
                ('joint_states', '/robot/joint_states'),
                ('joint_commands', '/robot/joint_commands'),
            ],
            output='screen'
        ),

        # Balance controller node
        Node(
            package='humanoid_control',
            executable='balance_node',
            name='balance_controller',
            parameters=[
                {'control_frequency': 100},
                {'stability_threshold': 0.1},
            ],
            output='screen'
        ),
    ])
```

## 6. Testing and Quality Assurance

### Unit Testing with pytest
```python
# test/test_controller.py
import unittest
import rclpy
from humanoid_control.controller import HumanoidController

class TestHumanoidController(unittest.TestCase):
    def setUp(self):
        rclpy.init()
        self.controller = HumanoidController()

    def tearDown(self):
        self.controller.destroy_node()
        rclpy.shutdown()

    def test_compute_joint_commands(self):
        # Test the joint command computation
        cmd = self.controller.compute_joint_commands()
        self.assertIsNotNone(cmd)
        self.assertEqual(len(cmd.position), len(self.controller.current_joint_states.name))

if __name__ == '__main__':
    unittest.main()
```

## 7. Best Practices for Humanoid Robotics Packages

### Modularity
- Separate concerns: control, perception, planning, and communication
- Create reusable libraries for common operations
- Use composition over inheritance

### Performance
- Minimize message copying
- Use appropriate QoS settings for real-time requirements
- Profile code for bottlenecks

### Safety
- Implement proper error handling
- Add safety limits and constraints
- Include emergency stop functionality

### Documentation
- Document all public interfaces
- Include example usage
- Maintain README files

## 8. Package Dependencies and Management

### Managing Dependencies
```xml
<!-- In package.xml -->
<depend>rclpy</depend>
<depend>std_msgs</depend>
<depend>sensor_msgs</depend>
<!-- Additional dependencies -->
<depend>control_msgs</depend>
<depend>trajectory_msgs</depend>
<depend>builtin_interfaces</depend>

<!-- Python-specific dependencies -->
<exec_depend>python3-numpy</exec_depend>
<exec_depend>python3-scipy</exec_depend>
<exec_depend>python3-transforms3d</exec_depend>
```

## Weekly Schedule Focus (Weeks 3-5)
During Weeks 3-5, we will focus on:
- Building ROS 2 packages with Python
- Launch files and parameter management
- Testing and quality assurance for robot packages
- Integration with humanoid robot control systems

## Resources
- [ROS 2 Package Development](https://docs.ros.org/en/rolling/Tutorials/Beginner-Client-Libraries/Creating-Your-First-ROS2-Package.html)
- [ament_python Build Type](https://github.com/ament/ament_python)
- [ROS 2 Launch Files](https://docs.ros.org/en/rolling/Tutorials/Intermediate/Launch/Creating-Launch-Files.html)
- [ROS 2 Testing](https://docs.ros.org/en/rolling/Tutorials/Testing/Using-Ament-Testing.html)
