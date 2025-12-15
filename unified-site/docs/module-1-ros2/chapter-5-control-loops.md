# Chapter 5: Control Loops for Humanoid Robots

## Overview
This chapter covers the implementation of control loops for humanoid robots using ROS 2. Students will learn to design and implement various control strategies, from low-level joint control to high-level whole-body control, ensuring stable and responsive robot behavior.

## Learning Objectives
By the end of this chapter, students will be able to:
- Implement different types of control loops for humanoid robots
- Design PID controllers for joint position, velocity, and effort control
- Implement whole-body control strategies for balance and locomotion
- Understand real-time constraints and timing considerations
- Integrate control systems with ROS 2's real-time capabilities

## 1. Introduction to Control Systems in Humanoid Robotics

Control systems are fundamental to humanoid robotics, managing everything from individual joint positions to complex whole-body behaviors like walking and balancing. Humanoid robots require multiple control layers:

- **Joint-level control**: Individual actuator positioning
- **Task-level control**: Coordinated movement of multiple joints
- **Whole-body control**: Balance and locomotion management
- **High-level control**: Behavior and planning integration

## 2. Basic Control Loop Structure

### Simple Control Loop with rclpy
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import time

class JointController(Node):
    def __init__(self):
        super().__init__('joint_controller')

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10
        )

        # Publishers
        self.joint_cmd_pub = self.create_publisher(
            JointTrajectory, 'joint_trajectory', 10
        )

        # Control loop timer (100Hz)
        self.control_timer = self.create_timer(0.01, self.control_loop)

        # Robot state
        self.current_positions = {}
        self.desired_positions = {}

        # PID parameters
        self.kp = 10.0
        self.ki = 0.1
        self.kd = 0.5

        # PID internal variables
        self.prev_error = {}
        self.integral_error = {}

        self.get_logger().info('Joint Controller initialized')

    def joint_state_callback(self, msg):
        """Update current joint positions"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_positions[name] = msg.position[i]

    def control_loop(self):
        """Main control loop executing at 100Hz"""
        # Calculate control commands for each joint
        commands = JointTrajectory()
        commands.joint_names = list(self.current_positions.keys())

        trajectory_point = JointTrajectoryPoint()
        trajectory_point.positions = []
        trajectory_point.velocities = []
        trajectory_point.accelerations = []

        for joint_name in commands.joint_names:
            if joint_name in self.desired_positions:
                current_pos = self.current_positions.get(joint_name, 0.0)
                desired_pos = self.desired_positions[joint_name]

                # Calculate PID control output
                control_output = self.compute_pid_output(joint_name, desired_pos, current_pos)

                # Add to trajectory
                trajectory_point.positions.append(control_output)
                trajectory_point.velocities.append(0.0)  # Simplified
                trajectory_point.accelerations.append(0.0)

        # Set timing
        trajectory_point.time_from_start = Duration(sec=0, nanosec=10000000)  # 10ms

        commands.points.append(trajectory_point)
        self.joint_cmd_pub.publish(commands)

    def compute_pid_output(self, joint_name, desired, current):
        """Compute PID control output for a single joint"""
        error = desired - current

        # Initialize if needed
        if joint_name not in self.integral_error:
            self.integral_error[joint_name] = 0.0
            self.prev_error[joint_name] = 0.0

        # Update integral and derivative terms
        self.integral_error[joint_name] += error * 0.01  # dt = 0.01s
        derivative = (error - self.prev_error[joint_name]) / 0.01

        # Compute PID output
        output = (self.kp * error +
                 self.ki * self.integral_error[joint_name] +
                 self.kd * derivative)

        # Update for next iteration
        self.prev_error[joint_name] = error

        # Apply limits
        output = max(min(output, 100.0), -100.0)  # Limit to ±100

        return current + output * 0.01  # Simple position update (in reality, this would be more complex)

    def set_desired_position(self, joint_name, position):
        """Set desired position for a joint"""
        self.desired_positions[joint_name] = position

def main(args=None):
    rclpy.init(args=args)

    controller = JointController()

    # Example: Set some desired positions
    controller.set_desired_position('left_hip', 0.1)
    controller.set_desired_position('right_hip', 0.1)

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down controller')
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 3. Advanced Control Strategies

### Whole-Body Control Example
```python
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Vector3, Twist
from std_msgs.msg import Float64MultiArray

class WholeBodyController(Node):
    def __init__(self):
        super().__init__('whole_body_controller')

        # Subscribers for sensor data
        self.joint_state_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, 'imu_data', self.imu_callback, 10
        )

        # Publishers for commands
        self.joint_cmd_pub = self.create_publisher(
            JointState, 'joint_commands', 10
        )
        self.com_pub = self.create_publisher(
            Twist, 'center_of_mass', 10
        )

        # Control loop timer
        self.control_timer = self.create_timer(0.005, self.whole_body_control_loop)  # 200Hz

        # Robot state
        self.joint_positions = {}
        self.joint_velocities = {}
        self.imu_data = {'orientation': [0, 0, 0, 1], 'angular_velocity': [0, 0, 0], 'linear_acceleration': [0, 0, 0]}

        # Balance control parameters
        self.balance_gains = {
            'kp': np.array([100.0, 100.0, 100.0]),  # Position gains [x, y, z]
            'kd': np.array([10.0, 10.0, 10.0]),     # Velocity gains
        }

        # Desired center of mass position
        self.desired_com = np.array([0.0, 0.0, 0.8])  # [x, y, z] in meters

        self.get_logger().info('Whole Body Controller initialized')

    def joint_state_callback(self, msg):
        """Update joint state information"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]
            if i < len(msg.velocity):
                self.joint_velocities[name] = msg.velocity[i]

    def imu_callback(self, msg):
        """Update IMU data"""
        self.imu_data = {
            'orientation': [
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w
            ],
            'angular_velocity': [
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z
            ],
            'linear_acceleration': [
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z
            ]
        }

    def whole_body_control_loop(self):
        """Main whole-body control loop"""
        # Calculate current center of mass (simplified)
        current_com = self.estimate_center_of_mass()

        # Calculate balance control commands
        balance_commands = self.compute_balance_control(current_com)

        # Calculate joint commands based on balance control
        joint_commands = self.compute_joint_commands(balance_commands)

        # Publish commands
        self.publish_joint_commands(joint_commands)
        self.publish_com_state(current_com)

    def estimate_center_of_mass(self):
        """Estimate center of mass position (simplified)"""
        # In reality, this would use forward kinematics and mass distribution
        # For now, return a simple approximation based on joint positions
        return np.array([0.0, 0.0, 0.8])  # Placeholder

    def compute_balance_control(self, current_com):
        """Compute balance control commands"""
        # Calculate error from desired center of mass
        com_error = self.desired_com - current_com

        # Apply PD control for balance
        balance_cmd = (self.balance_gains['kp'] * com_error[:3] +
                      self.balance_gains['kd'] * np.zeros(3))  # Velocity term would come from actual velocity

        return balance_cmd

    def compute_joint_commands(self, balance_commands):
        """Convert balance commands to joint commands"""
        # This would involve inverse kinematics and whole-body control algorithms
        # For now, return a simple joint command structure
        joint_cmd = JointState()
        joint_cmd.name = list(self.joint_positions.keys())
        joint_cmd.position = [0.0] * len(joint_cmd.name)  # Placeholder

        # Apply some simple balance corrections
        for i, joint_name in enumerate(joint_cmd.name):
            if 'hip' in joint_name:
                joint_cmd.position[i] = balance_commands[1] * 0.1  # Yaw correction
            elif 'ankle' in joint_name:
                joint_cmd.position[i] = -balance_commands[1] * 0.1  # Opposite correction

        return joint_cmd

    def publish_joint_commands(self, joint_cmd):
        """Publish joint commands"""
        self.joint_cmd_pub.publish(joint_cmd)

    def publish_com_state(self, com_pos):
        """Publish center of mass state"""
        com_msg = Twist()
        com_msg.linear.x = com_pos[0]
        com_msg.linear.y = com_pos[1]
        com_msg.linear.z = com_pos[2]
        self.com_pub.publish(com_msg)

def main(args=None):
    rclpy.init(args=args)
    controller = WholeBodyController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down whole body controller')
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 4. Timing and Real-Time Considerations

### Real-Time Control with Proper Timing
```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from std_msgs.msg import Header
import time
from threading import Thread
import numpy as np

class RealTimeController(Node):
    def __init__(self):
        super().__init__('real_time_controller')

        # Create a publisher with real-time QoS
        qos_profile = QoSProfile(depth=1)
        qos_profile.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL

        self.control_cmd_pub = self.create_publisher(
            Float64MultiArray, 'real_time_commands', qos_profile
        )

        # Control parameters
        self.control_frequency = 1000  # 1kHz
        self.dt = 1.0 / self.control_frequency

        # Start real-time control thread
        self.control_thread = Thread(target=self.real_time_control_loop)
        self.control_thread.daemon = True
        self.running = True

        self.get_logger().info(f'Starting real-time controller at {self.control_frequency}Hz')

    def start_control(self):
        """Start the real-time control loop"""
        self.control_thread.start()

    def real_time_control_loop(self):
        """Real-time control loop with precise timing"""
        last_time = time.time()

        while self.running:
            current_time = time.time()
            elapsed = current_time - last_time

            if elapsed >= self.dt:
                # Execute control computation
                self.execute_control_step()

                # Publish commands
                self.publish_commands()

                # Update timing
                last_time = current_time
            else:
                # Sleep for remaining time to maintain frequency
                time.sleep(max(0, self.dt - elapsed))

    def execute_control_step(self):
        """Execute one step of control computation"""
        # Placeholder for control algorithm
        # This should contain the actual control computation
        pass

    def publish_commands(self):
        """Publish control commands"""
        cmd_msg = Float64MultiArray()
        cmd_msg.data = [0.0] * 20  # Example: 20 joint commands
        self.control_cmd_pub.publish(cmd_msg)

    def stop_control(self):
        """Stop the real-time control loop"""
        self.running = False
        if self.control_thread.is_alive():
            self.control_thread.join()

def main(args=None):
    rclpy.init(args=args)
    controller = RealTimeController()

    try:
        controller.start_control()
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down real-time controller')
    finally:
        controller.stop_control()
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 5. Control Architecture Patterns

### Hierarchical Control Architecture
```python
import rclpy
from rclpy.node import Node
from enum import Enum

class ControlMode(Enum):
    IDLE = 0
    JOINT_POSITION = 1
    JOINT_VELOCITY = 2
    JOINT_EFFORT = 3
    TRAJECTORY = 4
    WHOLE_BODY = 5

class HierarchicalController(Node):
    def __init__(self):
        super().__init__('hierarchical_controller')

        # Control mode
        self.control_mode = ControlMode.IDLE

        # Initialize different control layers
        self.joint_controller = JointControllerLayer(self)
        self.task_controller = TaskControllerLayer(self)
        self.whole_body_controller = WholeBodyControllerLayer(self)

        # Control loop
        self.control_timer = self.create_timer(0.01, self.hierarchical_control_loop)

        self.get_logger().info('Hierarchical Controller initialized')

    def hierarchical_control_loop(self):
        """Execute control based on current mode"""
        if self.control_mode == ControlMode.JOINT_POSITION:
            self.joint_controller.execute()
        elif self.control_mode == ControlMode.TASK:
            self.task_controller.execute()
        elif self.control_mode == ControlMode.WHOLE_BODY:
            self.whole_body_controller.execute()
        # Add more modes as needed

class JointControllerLayer:
    def __init__(self, parent_node):
        self.node = parent_node
        # Initialize joint-specific control parameters

    def execute(self):
        # Execute joint-level control
        pass

class TaskControllerLayer:
    def __init__(self, parent_node):
        self.node = parent_node
        # Initialize task-specific control parameters

    def execute(self):
        # Execute task-level control
        pass

class WholeBodyControllerLayer:
    def __init__(self, parent_node):
        self.node = parent_node
        # Initialize whole-body control parameters

    def execute(self):
        # Execute whole-body control
        pass
```

## 6. Safety and Error Handling

### Safe Control Implementation
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
import numpy as np

class SafeController(Node):
    def __init__(self):
        super().__init__('safe_controller')

        # Publishers and subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10
        )
        self.emergency_stop_pub = self.create_publisher(Bool, 'emergency_stop', 10)

        # Safety parameters
        self.joint_limits = {
            'left_hip': (-1.5, 1.5),
            'right_hip': (-1.5, 1.5),
            'left_knee': (0.0, 2.5),
            'right_knee': (0.0, 2.5),
            # Add more joints as needed
        }

        self.max_velocity = 5.0  # rad/s
        self.max_effort = 100.0  # Nm

        # Control loop
        self.control_timer = self.create_timer(0.01, self.safe_control_loop)

        self.current_positions = {}
        self.current_velocities = {}
        self.emergency_stop_active = False

        self.get_logger().info('Safe Controller initialized')

    def joint_state_callback(self, msg):
        """Update joint states with safety checks"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                # Check position limits
                if name in self.joint_limits:
                    min_pos, max_pos = self.joint_limits[name]
                    if not (min_pos <= msg.position[i] <= max_pos):
                        self.get_logger().warn(f'Joint {name} position limit exceeded: {msg.position[i]}')
                        self.activate_emergency_stop()
                        return

                self.current_positions[name] = msg.position[i]

            if i < len(msg.velocity):
                # Check velocity limits
                if abs(msg.velocity[i]) > self.max_velocity:
                    self.get_logger().warn(f'Joint {name} velocity limit exceeded: {msg.velocity[i]}')
                    self.activate_emergency_stop()
                    return

                self.current_velocities[name] = msg.velocity[i]

    def safe_control_loop(self):
        """Main control loop with safety checks"""
        if self.emergency_stop_active:
            self.publish_emergency_stop()
            return

        # Perform safety checks before control execution
        if not self.perform_safety_checks():
            self.activate_emergency_stop()
            return

        # Execute control algorithm
        self.execute_control()

    def perform_safety_checks(self):
        """Perform various safety checks"""
        # Check for NaN values
        for pos in self.current_positions.values():
            if np.isnan(pos):
                self.get_logger().error('NaN detected in joint positions')
                return False

        for vel in self.current_velocities.values():
            if np.isnan(vel):
                self.get_logger().error('NaN detected in joint velocities')
                return False

        # Check for extreme values
        for name, vel in self.current_velocities.items():
            if abs(vel) > self.max_velocity * 2:  # 2x safety margin
                self.get_logger().error(f'Extreme velocity detected for {name}: {vel}')
                return False

        return True

    def execute_control(self):
        """Execute the main control algorithm"""
        # Placeholder for control algorithm
        pass

    def activate_emergency_stop(self):
        """Activate emergency stop"""
        self.emergency_stop_active = True
        self.get_logger().error('EMERGENCY STOP ACTIVATED')
        self.publish_emergency_stop()

    def publish_emergency_stop(self):
        """Publish emergency stop command"""
        stop_msg = Bool()
        stop_msg.data = True
        self.emergency_stop_pub.publish(stop_msg)

def main(args=None):
    rclpy.init(args=args)
    controller = SafeController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down safe controller')
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 7. Performance Optimization

### Optimized Control Loop
```python
import rclpy
from rclpy.node import Node
import numpy as np
from collections import deque
import time

class OptimizedController(Node):
    def __init__(self):
        super().__init__('optimized_controller')

        # Use numpy arrays for efficient computation
        self.joint_positions = np.array([])
        self.joint_velocities = np.array([])
        self.desired_positions = np.array([])

        # Circular buffer for performance tracking
        self.control_cycle_times = deque(maxlen=100)

        # Control loop with performance monitoring
        self.control_timer = self.create_timer(0.001, self.optimized_control_loop)  # 1kHz

        self.get_logger().info('Optimized Controller initialized')

    def optimized_control_loop(self):
        """Optimized control loop with performance monitoring"""
        start_time = time.perf_counter()

        # Efficient control computation using numpy
        if len(self.joint_positions) > 0 and len(self.desired_positions) > 0:
            # Vectorized error calculation
            position_error = self.desired_positions - self.joint_positions

            # Vectorized control law (simplified PD)
            control_output = 10.0 * position_error - 1.0 * self.joint_velocities

            # Apply limits efficiently
            control_output = np.clip(control_output, -100.0, 100.0)

        # Track performance
        end_time = time.perf_counter()
        cycle_time = (end_time - start_time) * 1000  # ms
        self.control_cycle_times.append(cycle_time)

        # Log performance if needed
        if len(self.control_cycle_times) == 100:  # Buffer full
            avg_time = sum(self.control_cycle_times) / len(self.control_cycle_times)
            if avg_time > 0.8:  # 80% of cycle time
                self.get_logger().warn(f'Control loop taking {avg_time:.2f}ms, approaching deadline')
```

## Weekly Schedule Focus (Weeks 3-5)
During Weeks 3-5, we will focus on:
- Implementing control loops with proper timing
- Designing PID controllers for robot joints
- Whole-body control strategies for humanoid robots
- Safety considerations in control systems

## Resources
- [ROS 2 Control](https://control.ros.org/)
- [Real-time Control Systems](https://en.wikipedia.org/wiki/Real-time_control_systems)
- [PID Controller Theory](https://en.wikipedia.org/wiki/PID_controller)
- [Whole Body Control](https://humanoid-walk.readthedocs.io/)
