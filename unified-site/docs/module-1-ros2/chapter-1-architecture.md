# Chapter 1: ROS 2 Architecture

## Overview
This chapter introduces the fundamental architecture of ROS 2 (Robot Operating System 2), which serves as the nervous system for humanoid robots. ROS 2 provides the middleware for robot control, enabling communication between different software components and hardware interfaces.

## Learning Objectives
By the end of this chapter, students will be able to:
- Understand the core concepts of ROS 2 architecture
- Identify and explain the main components of ROS 2
- Implement basic ROS 2 nodes, topics, and services
- Bridge Python agents to ROS controllers using rclpy
- Work with URDF (Unified Robot Description Format) for humanoid robots

## 1. Introduction to ROS 2

ROS 2 is a flexible framework for writing robot software, designed to address the challenges of building complex robotic systems. Unlike its predecessor, ROS 2 provides improved security, real-time capabilities, and better support for distributed systems, making it ideal for humanoid robotics applications.

### Key Improvements over ROS 1
- **Real-time support**: Enhanced capabilities for time-critical applications
- **Security**: Built-in security features for safe robot operation
- **Distributed architecture**: Better support for multi-robot systems
- **Professional-grade middleware**: Based on DDS (Data Distribution Service)

## 2. Core Concepts

### Nodes
Nodes are the fundamental building blocks of ROS 2 applications. Each node performs a specific function and communicates with other nodes through topics, services, and actions.

```python
import rclpy
from rclpy.node import Node

class HumanoidControllerNode(Node):
    def __init__(self):
        super().__init__('humanoid_controller')
        self.get_logger().info('Humanoid Controller Node Started')
```

### Topics and Message Passing
Topics enable asynchronous communication between nodes using a publish-subscribe pattern. This is ideal for continuous data streams like sensor readings or actuator commands.

### Services
Services provide synchronous request-response communication, useful for operations that require immediate responses, such as configuration changes or specific actions.

### Actions
Actions are used for long-running tasks that may provide feedback during execution, perfect for complex humanoid movements that take time to complete.

## 3. ROS 2 Middleware Architecture

ROS 2 uses DDS (Data Distribution Service) as its underlying communication middleware. This provides:

- **Discovery**: Automatic discovery of nodes on the network
- **Transport**: Reliable message delivery with configurable QoS (Quality of Service) settings
- **Data modeling**: Standardized message formats for interoperability

## 4. Working with rclpy

rclpy is the Python client library for ROS 2, enabling Python-based robot applications. It provides the bridge between Python agents and ROS controllers.

### Example: Creating a Publisher Node
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class HumanoidSensorNode(Node):
    def __init__(self):
        super().__init__('humanoid_sensor_publisher')
        self.publisher_ = self.create_publisher(String, 'sensor_data', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Sensor reading: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1
```

## 5. URDF (Unified Robot Description Format)

URDF is essential for describing humanoid robots in ROS 2. It defines the robot's physical structure, including:

- **Links**: Rigid parts of the robot (e.g., torso, limbs)
- **Joints**: Connections between links (e.g., hinges, prismatic joints)
- **Visual and collision properties**: Appearance and physics properties
- **Inertial properties**: Mass, center of mass, and moments of inertia

### Example URDF Structure for Humanoid
```xml
<?xml version="1.0"?>
<robot name="humanoid_robot">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.2 0.2"/>
      </geometry>
    </visual>
  </link>

  <!-- Torso -->
  <link name="torso">
    <visual>
      <geometry>
        <box size="0.2 0.5 0.2"/>
      </geometry>
    </visual>
  </link>

  <!-- Joint connecting base to torso -->
  <joint name="base_to_torso" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
  </joint>
</robot>
```

## 6. Launch Files and Parameter Management

Launch files allow you to start multiple nodes simultaneously with specific configurations:

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='humanoid_control',
            executable='controller',
            name='humanoid_controller',
            parameters=[
                {'kp': 1.0},
                {'ki': 0.1},
                {'kd': 0.05}
            ]
        )
    ])
```

## 7. Best Practices for Humanoid Robotics

- **Modularity**: Design nodes to perform specific functions
- **Error handling**: Implement robust error handling for safety
- **Real-time considerations**: Use appropriate QoS settings for time-critical operations
- **Security**: Implement ROS 2 security features for safe operation
- **Testing**: Develop comprehensive tests for each component

## Weekly Schedule Focus (Weeks 3-5)
During Weeks 3-5, we will focus on:
- ROS 2 architecture and core concepts
- Building ROS 2 packages with Python
- Launch files and parameter management
- Integration with humanoid robot control systems

## Resources
- [ROS 2 Documentation](https://docs.ros.org/en/rolling/)
- [rclpy API Documentation](https://docs.ros.org/en/rolling/p/rclpy/)
- [URDF Tutorials](http://wiki.ros.org/urdf/Tutorials)
- [ROS 2 Design](https://design.ros2.org/)
