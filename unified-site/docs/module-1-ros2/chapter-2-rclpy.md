# Chapter 2: Bridging Python Agents to ROS Controllers with rclpy

## Overview
This chapter focuses on using rclpy, the Python client library for ROS 2, to bridge Python-based AI agents with ROS controllers. This integration is crucial for humanoid robotics, where Python-based machine learning models need to communicate with real-time robot control systems.

## Learning Objectives
By the end of this chapter, students will be able to:
- Understand the role of rclpy in ROS 2 Python applications
- Create Python nodes that interface with robot controllers
- Implement message passing between Python agents and ROS systems
- Design control loops that integrate AI decision-making with robot actuation
- Deploy Python-based controllers to edge computing platforms like NVIDIA Jetson

## 1. Introduction to rclpy

rclpy is the official Python client library for ROS 2, providing a Python API for ROS 2 concepts such as nodes, publishers, subscribers, services, and actions. It serves as the bridge between Python-based AI algorithms and the ROS 2 ecosystem.

### Why rclpy for Humanoid Robotics?
- **AI Integration**: Python is the dominant language for AI and machine learning
- **Rapid Prototyping**: Faster development cycles for complex humanoid behaviors
- **Community Support**: Extensive libraries for computer vision, NLP, and control
- **Cross-platform**: Runs on both development machines and edge devices

## 2. Setting Up rclpy Nodes

### Basic Node Structure
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from control_msgs.msg import JointTrajectoryControllerState

class PythonAIController(Node):
    def __init__(self):
        super().__init__('python_ai_controller')

        # Publishers for sending commands to robot
        self.joint_command_publisher = self.create_publisher(
            JointState, 'joint_commands', 10
        )

        # Subscribers for receiving sensor data
        self.sensor_subscriber = self.create_subscription(
            JointTrajectoryControllerState,
            'joint_states',
            self.joint_state_callback,
            10
        )

        # Timer for control loop
        self.control_timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('Python AI Controller initialized')

    def joint_state_callback(self, msg):
        # Process incoming joint state data
        self.get_logger().info(f'Received joint states: {len(msg.joint_names)} joints')

    def control_loop(self):
        # AI-driven control logic goes here
        self.get_logger().info('Executing control loop')
```

## 3. Bridging AI Agents to Robot Controllers

### Example: AI Decision-Making Node
```python
import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class AIBasedController(Node):
    def __init__(self):
        super().__init__('ai_based_controller')

        # Publishers and subscribers
        self.trajectory_publisher = self.create_publisher(
            JointTrajectory, 'joint_trajectory', 10
        )

        self.vision_subscriber = self.create_subscription(
            String, 'vision_output', self.vision_callback, 10
        )

        self.imu_subscriber = self.create_subscription(
            JointState, 'imu_data', self.imu_callback, 10
        )

        # AI model parameters (simplified for example)
        self.balance_weights = np.random.rand(12, 6)  # Example weights
        self.target_position = None

        self.get_logger().info('AI-Based Controller initialized')

    def vision_callback(self, msg):
        # Process vision data and make decisions
        vision_data = msg.data
        self.get_logger().info(f'Vision data: {vision_data}')

        # AI decision-making based on vision
        if 'object' in vision_data:
            self.plan_grasp_action(vision_data)

    def imu_callback(self, msg):
        # Process IMU data for balance control
        imu_data = np.array(msg.position)  # Simplified
        balance_correction = self.calculate_balance(imu_data)

        # Publish balance correction commands
        self.publish_balance_correction(balance_correction)

    def calculate_balance(self, imu_data):
        # AI-based balance calculation
        correction = np.dot(self.balance_weights, imu_data[:6])
        return correction

    def plan_grasp_action(self, vision_data):
        # Plan grasping trajectory based on vision data
        self.get_logger().info('Planning grasp action')
        # Implementation would involve path planning algorithms

    def publish_balance_correction(self, correction):
        # Publish balance correction commands to robot
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = ['left_hip', 'right_hip', 'left_ankle', 'right_ankle']  # Example

        point = JointTrajectoryPoint()
        point.positions = correction.tolist()
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 100000000  # 100ms

        trajectory_msg.points.append(point)
        self.trajectory_publisher.publish(trajectory_msg)

## 4. Advanced rclpy Patterns for Humanoid Control

### Asynchronous Service Calls
```python
import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from control_msgs.action import FollowJointTrajectory

class AdvancedAIController(Node):
    def __init__(self):
        super().__init__('advanced_ai_controller')

        # Use a separate callback group for service calls
        self.service_cb_group = MutuallyExclusiveCallbackGroup()

        # Action client for trajectory execution
        self.trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            'joint_trajectory_controller/follow_joint_trajectory'
        )

    def send_trajectory_goal(self, trajectory):
        # Wait for action server
        if not self.trajectory_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error('Trajectory server not available')
            return

        # Send goal
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = trajectory

        self.trajectory_client.send_goal_async(
            goal_msg,
            feedback_callback=self.trajectory_feedback_callback
        )

    def trajectory_feedback_callback(self, feedback_msg):
        # Handle trajectory execution feedback
        self.get_logger().info(f'Trajectory progress: {feedback_msg.feedback}')
```

## 5. Performance Considerations

When bridging Python AI agents to real-time controllers, consider:

- **Timing constraints**: Ensure AI processing doesn't exceed control loop timing
- **Memory management**: Monitor memory usage during AI inference
- **Threading**: Use appropriate threading models for parallel processing
- **QoS settings**: Configure Quality of Service for reliable communication

## 6. Edge Deployment Considerations

For deployment on edge devices like NVIDIA Jetson:

- Optimize Python code for resource-constrained environments
- Use efficient data structures and minimize memory allocations
- Consider using compiled Python extensions for performance-critical sections
- Implement proper error handling and recovery mechanisms

## Weekly Schedule Focus (Weeks 3-5)
During Weeks 3-5, we will focus on:
- Building ROS 2 packages with Python
- Implementing control loops with rclpy
- Integrating AI agents with robot controllers
- Testing on simulation and hardware platforms

## Resources
- [rclpy Documentation](https://docs.ros.org/en/rolling/p/rclpy/)
- [ROS 2 Python Tutorials](https://docs.ros.org/en/rolling/Tutorials/Beginner-Client-Libraries/Using-Subscriptions-Python.html)
- [NVIDIA Jetson ROS Development](https://nvidia-isaac-ros.github.io/)
- [Python Robotics Libraries](https://pypi.org/search/?q=robotics)
