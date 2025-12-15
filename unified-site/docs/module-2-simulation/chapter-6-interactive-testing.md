# Chapter 6: Interactive Testing

## Overview
This chapter covers interactive testing methodologies for humanoid robots in simulation environments. Students will learn to create comprehensive test scenarios, implement debugging tools, and use visualization techniques to validate robot behavior in real-time. The focus is on developing efficient testing workflows that bridge the gap between simulation and real-world deployment.

## Learning Objectives
By the end of this chapter, students will be able to:
- Design and implement interactive test scenarios for humanoid robots
- Create debugging and visualization tools for real-time monitoring
- Develop comprehensive test suites that validate robot functionality
- Implement automated testing procedures with manual oversight
- Use interactive testing to identify and resolve robot behavior issues
- Validate sim-to-real transfer through systematic testing

## 1. Introduction to Interactive Testing for Humanoid Robots

Interactive testing is a critical component of humanoid robot development that allows developers to:
- **Validate complex behaviors** in real-time simulation
- **Debug control algorithms** with immediate feedback
- **Test safety protocols** in a controlled environment
- **Optimize performance** through iterative testing
- **Validate sim-to-real transfer** before hardware deployment

### Key Components of Interactive Testing:
- **Real-time visualization** of robot state and sensor data
- **Interactive control interfaces** for manual intervention
- **Automated test scenarios** with manual oversight
- **Performance monitoring** and logging tools
- **Scenario replay** and analysis capabilities

## 2. Interactive Test Environment Setup

### Creating Test Worlds
```xml
<!-- worlds/interactive_test.world -->
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="interactive_test_world">
    <!-- Include standard models -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Test obstacles and markers -->
    <model name="test_obstacle_1">
      <pose>2 0 0.5 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry>
            <box>
              <size>0.5 0.5 1.0</size>
            </box>
          </geometry>
          <material>
            <ambient>1 0 0 1</ambient>
            <diffuse>1 0 0 1</diffuse>
          </material>
        </visual>
        <collision name="collision">
          <geometry>
            <box>
              <size>0.5 0.5 1.0</size>
            </box>
          </geometry>
        </collision>
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.083</ixx>
            <iyy>0.083</iyy>
            <izz>0.083</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <!-- Test markers for navigation -->
    <model name="test_marker_1">
      <static>true</static>
      <pose>3 2 0.1 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>0.1</radius>
              <length>0.2</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>0 1 0 1</ambient>
            <diffuse>0 1 0 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Physics configuration -->
    <physics name="test_physics" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_update_rate>1000</real_time_update_rate>
      <real_time_factor>1.0</real_time_factor>
      <ode>
        <solver>
          <type>quick</type>
          <iters>100</iters>
          <sor>1.3</sor>
        </solver>
        <constraints>
          <cfm>0.000001</cfm>
          <erp>0.2</erp>
        </constraints>
      </ode>
    </physics>
  </world>
</sdf>
```

### Interactive Test Launch File
```python
# launch/interactive_test.launch.py
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
import os

def generate_launch_description():
    # Launch configuration
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    robot_model = LaunchConfiguration('robot_model', default='humanoid')

    # Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': PathJoinSubstitution([
                FindPackageShare('humanoid_simulation'),
                'worlds',
                'interactive_test.world'
            ]),
            'use_sim_time': use_sim_time
        }.items()
    )

    # Robot spawn node
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'humanoid_robot',
            '-x', '0', '-y', '0', '-z', '1.0'
        ],
        output='screen'
    )

    # Interactive test interface
    test_interface = Node(
        package='humanoid_simulation',
        executable='interactive_test_interface',
        name='test_interface',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_model': robot_model
        }],
        output='screen'
    )

    # RViz for visualization
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=[
            '-d', PathJoinSubstitution([
                FindPackageShare('humanoid_simulation'),
                'rviz',
                'interactive_test.rviz'
            ])
        ],
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation clock if true'
        ),
        DeclareLaunchArgument(
            'robot_model',
            default_value='humanoid',
            description='Robot model to use for testing'
        ),
        gazebo,
        spawn_entity,
        test_interface,
        rviz
    ])
```

## 3. Real-time Debugging and Visualization Tools

### Robot State Visualization Node
```python
# interactive_test_interface.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import Float32, String
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import numpy as np
import math

class InteractiveTestInterface(Node):
    def __init__(self):
        super().__init__('interactive_test_interface')

        # Robot state storage
        self.joint_states = JointState()
        self.imu_data = Imu()
        self.robot_pose = Vector3()
        self.robot_twist = Twist()

        # Publishers for visualization
        self.marker_pub = self.create_publisher(MarkerArray, 'test_markers', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Subscribers for robot data
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )

        # Interactive control subscribers
        self.test_command_sub = self.create_subscription(
            String, '/test_commands', self.test_command_callback, 10
        )

        # Timer for visualization updates
        self.viz_timer = self.create_timer(0.1, self.update_visualization)

        # Test scenario management
        self.current_scenario = "idle"
        self.test_scenarios = {
            "idle": self.scenario_idle,
            "walk_forward": self.scenario_walk_forward,
            "turn": self.scenario_turn,
            "balance_test": self.scenario_balance_test,
            "obstacle_avoidance": self.scenario_obstacle_avoidance
        }

        self.get_logger().info('Interactive Test Interface initialized')

    def joint_state_callback(self, msg):
        """Update joint state information"""
        self.joint_states = msg

    def imu_callback(self, msg):
        """Update IMU data"""
        self.imu_data = msg

    def test_command_callback(self, msg):
        """Handle test commands"""
        command = msg.data
        if command in self.test_scenarios:
            self.current_scenario = command
            self.get_logger().info(f'Switched to test scenario: {command}')
        else:
            self.get_logger().warn(f'Unknown test command: {command}')

    def update_visualization(self):
        """Update real-time visualization markers"""
        markers = MarkerArray()

        # Robot pose marker
        robot_marker = Marker()
        robot_marker.header.frame_id = "map"
        robot_marker.header.stamp = self.get_clock().now().to_msg()
        robot_marker.ns = "robot"
        robot_marker.id = 0
        robot_marker.type = Marker.CUBE
        robot_marker.action = Marker.ADD
        robot_marker.pose.position = self.robot_pose
        robot_marker.pose.orientation.w = 1.0
        robot_marker.scale.x = 0.5
        robot_marker.scale.y = 0.3
        robot_marker.scale.z = 1.0
        robot_marker.color.r = 0.0
        robot_marker.color.g = 0.0
        robot_marker.color.b = 1.0
        robot_marker.color.a = 0.8
        markers.markers.append(robot_marker)

        # Joint state visualization
        for i, (name, position) in enumerate(zip(self.joint_states.name, self.joint_states.position)):
            if 'joint' in name:  # Only visualize joint-related markers
                joint_marker = Marker()
                joint_marker.header.frame_id = "base_link"
                joint_marker.header.stamp = self.get_clock().now().to_msg()
                joint_marker.ns = "joints"
                joint_marker.id = i + 1
                joint_marker.type = Marker.SPHERE
                joint_marker.action = Marker.ADD
                joint_marker.pose.position.x = math.cos(position) * 0.2
                joint_marker.pose.position.y = math.sin(position) * 0.2
                joint_marker.pose.position.z = 0.1 * i
                joint_marker.pose.orientation.w = 1.0
                joint_marker.scale.x = 0.05
                joint_marker.scale.y = 0.05
                joint_marker.scale.z = 0.05
                joint_marker.color.r = 1.0
                joint_marker.color.g = 1.0
                joint_marker.color.b = 0.0
                joint_marker.color.a = 0.8
                markers.markers.append(joint_marker)

        # Publish markers
        self.marker_pub.publish(markers)

        # Broadcast transforms
        self.broadcast_transforms()

    def broadcast_transforms(self):
        """Broadcast robot transforms for visualization"""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'base_link'
        t.transform.translation = self.robot_pose
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)

    def scenario_idle(self):
        """Idle scenario - no specific actions"""
        pass

    def scenario_walk_forward(self):
        """Walking forward test scenario"""
        # This would send walking commands to the robot
        self.get_logger().info('Executing walk forward scenario')

    def scenario_turn(self):
        """Turning test scenario"""
        self.get_logger().info('Executing turn scenario')

    def scenario_balance_test(self):
        """Balance test scenario"""
        # Monitor IMU data for balance
        roll, pitch, yaw = self.get_euler_from_quaternion([
            self.imu_data.orientation.x,
            self.imu_data.orientation.y,
            self.imu_data.orientation.z,
            self.imu_data.orientation.w
        ])

        if abs(roll) > 0.5 or abs(pitch) > 0.5:
            self.get_logger().warn(f'Balance threshold exceeded: roll={roll:.2f}, pitch={pitch:.2f}')

    def scenario_obstacle_avoidance(self):
        """Obstacle avoidance test scenario"""
        self.get_logger().info('Executing obstacle avoidance scenario')

    def get_euler_from_quaternion(self, quaternion):
        """Convert quaternion to Euler angles"""
        x, y, z, w = quaternion

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

def main(args=None):
    rclpy.init(args=args)
    test_interface = InteractiveTestInterface()

    try:
        rclpy.spin(test_interface)
    except KeyboardInterrupt:
        test_interface.get_logger().info('Shutting down interactive test interface')
    finally:
        test_interface.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 4. Automated Test Scenarios with Manual Oversight

### Test Scenario Manager
```python
# test_scenario_manager.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32
from geometry_msgs.msg import Pose, Twist
from builtin_interfaces.msg import Time
import time
import json
from datetime import datetime

class TestScenarioManager(Node):
    def __init__(self):
        super().__init__('test_scenario_manager')

        # Test management
        self.test_scenarios = {}
        self.current_test = None
        self.test_results = {}
        self.test_history = []

        # Publishers and subscribers
        self.test_command_pub = self.create_publisher(String, '/test_commands', 10)
        self.test_status_pub = self.create_publisher(String, '/test_status', 10)
        self.test_result_pub = self.create_publisher(String, '/test_results', 10)

        # Subscribers for test feedback
        self.test_feedback_sub = self.create_subscription(
            String, '/test_feedback', self.test_feedback_callback, 10
        )

        # Timer for test management
        self.test_timer = self.create_timer(1.0, self.manage_tests)

        # Initialize test scenarios
        self.initialize_test_scenarios()

        self.get_logger().info('Test Scenario Manager initialized')

    def initialize_test_scenarios(self):
        """Initialize available test scenarios"""
        self.test_scenarios = {
            "basic_mobility": {
                "name": "Basic Mobility Test",
                "description": "Test basic walking and turning capabilities",
                "steps": [
                    {"command": "walk_forward", "duration": 5.0},
                    {"command": "turn", "duration": 3.0},
                    {"command": "walk_forward", "duration": 5.0}
                ],
                "success_criteria": ["no_fall", "reach_target", "stable_balance"],
                "priority": 1
            },
            "balance_stability": {
                "name": "Balance Stability Test",
                "description": "Test robot's ability to maintain balance",
                "steps": [
                    {"command": "balance_test", "duration": 10.0}
                ],
                "success_criteria": ["imu_stable", "no_excessive_tilt"],
                "priority": 2
            },
            "obstacle_navigation": {
                "name": "Obstacle Navigation Test",
                "description": "Test obstacle detection and avoidance",
                "steps": [
                    {"command": "obstacle_avoidance", "duration": 15.0}
                ],
                "success_criteria": ["avoid_obstacles", "reach_goal"],
                "priority": 3
            }
        }

    def start_test_scenario(self, scenario_name):
        """Start a specific test scenario"""
        if scenario_name not in self.test_scenarios:
            self.get_logger().error(f'Unknown test scenario: {scenario_name}')
            return False

        self.get_logger().info(f'Starting test scenario: {scenario_name}')
        self.current_test = {
            "name": scenario_name,
            "start_time": self.get_clock().now().nanoseconds / 1e9,
            "current_step": 0,
            "status": "running",
            "results": []
        }

        # Execute the first step
        self.execute_test_step(scenario_name, 0)
        return True

    def execute_test_step(self, scenario_name, step_index):
        """Execute a specific step in a test scenario"""
        scenario = self.test_scenarios[scenario_name]
        if step_index >= len(scenario["steps"]):
            self.complete_test_scenario(scenario_name)
            return

        step = scenario["steps"][step_index]
        command_msg = String()
        command_msg.data = step["command"]
        self.test_command_pub.publish(command_msg)

        self.get_logger().info(f'Executing step {step_index + 1}/{len(scenario["steps"])}: {step["command"]}')

        # Schedule next step
        self.current_test["current_step"] = step_index
        self.current_test["step_start_time"] = self.get_clock().now().nanoseconds / 1e9

        # Use timer to move to next step after duration
        timer = self.create_timer(step["duration"], lambda: self.next_test_step(scenario_name, step_index + 1))

    def next_test_step(self, scenario_name, next_step_index):
        """Move to the next test step"""
        if self.current_test and self.current_test["name"] == scenario_name:
            self.execute_test_step(scenario_name, next_step_index)

    def complete_test_scenario(self, scenario_name):
        """Complete a test scenario and record results"""
        if self.current_test:
            self.current_test["end_time"] = self.get_clock().now().nanoseconds / 1e9
            self.current_test["status"] = "completed"

            # Evaluate success criteria
            success = self.evaluate_success_criteria(scenario_name)
            self.current_test["success"] = success

            # Record test result
            self.test_results[scenario_name] = self.current_test.copy()
            self.test_history.append(self.current_test.copy())

            # Publish test results
            result_msg = String()
            result_msg.data = json.dumps({
                "test_name": scenario_name,
                "success": success,
                "duration": self.current_test["end_time"] - self.current_test["start_time"],
                "steps_completed": len(self.test_scenarios[scenario_name]["steps"])
            })
            self.test_result_pub.publish(result_msg)

            self.get_logger().info(f'Test scenario completed: {scenario_name}, Success: {success}')
            self.current_test = None

    def evaluate_success_criteria(self, scenario_name):
        """Evaluate if success criteria were met"""
        # This would typically involve checking sensor data and robot state
        # For now, we'll return True for demonstration
        return True

    def test_feedback_callback(self, msg):
        """Handle test feedback from other nodes"""
        try:
            feedback = json.loads(msg.data)
            if self.current_test:
                self.current_test["results"].append(feedback)
        except json.JSONDecodeError:
            self.get_logger().warn('Invalid test feedback format')

    def manage_tests(self):
        """Main test management loop"""
        # Check if we have a running test
        if self.current_test:
            # Check if test is taking too long
            current_time = self.get_clock().now().nanoseconds / 1e9
            elapsed = current_time - self.current_test["start_time"]

            if elapsed > 60:  # 1 minute timeout
                self.get_logger().warn('Test scenario timeout, stopping')
                self.stop_current_test()

        # Publish test status
        status_msg = String()
        if self.current_test:
            status_msg.data = f"Running: {self.current_test['name']}, Step: {self.current_test['current_step'] + 1}"
        else:
            status_msg.data = "No active test"
        self.test_status_pub.publish(status_msg)

    def stop_current_test(self):
        """Stop the current test scenario"""
        if self.current_test:
            self.current_test["status"] = "stopped"
            self.current_test = None
            self.get_logger().info('Current test scenario stopped')

def main(args=None):
    rclpy.init(args=args)
    test_manager = TestScenarioManager()

    try:
        rclpy.spin(test_manager)
    except KeyboardInterrupt:
        test_manager.get_logger().info('Shutting down test scenario manager')
    finally:
        test_manager.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 5. Performance Monitoring and Analysis

### Performance Monitor Node
```python
# performance_monitor.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Time
import time
import statistics
from collections import deque
import json

class PerformanceMonitor(Node):
    def __init__(self):
        super().__init__('performance_monitor')

        # Performance metrics storage
        self.metrics = {
            'control_loop_times': deque(maxlen=1000),
            'sensor_update_rates': deque(maxlen=1000),
            'joint_positions': {},
            'cpu_usage': deque(maxlen=1000),
            'memory_usage': deque(maxlen=1000)
        }

        # Publishers for performance data
        self.performance_pub = self.create_publisher(String, '/performance_metrics', 10)
        self.cpu_usage_pub = self.create_publisher(Float32, '/cpu_usage', 10)
        self.memory_usage_pub = self.create_publisher(Float32, '/memory_usage', 10)

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )

        # Timers for monitoring
        self.monitor_timer = self.create_timer(0.1, self.monitor_performance)
        self.report_timer = self.create_timer(5.0, self.report_performance)

        # Track control loop performance
        self.last_control_time = time.time()

        self.get_logger().info('Performance Monitor initialized')

    def joint_state_callback(self, msg):
        """Monitor joint state updates"""
        current_time = time.time()
        loop_time = current_time - self.last_control_time
        self.metrics['control_loop_times'].append(loop_time)
        self.last_control_time = current_time

        # Store joint positions
        for name, position in zip(msg.name, msg.position):
            if name not in self.metrics['joint_positions']:
                self.metrics['joint_positions'][name] = deque(maxlen=100)
            self.metrics['joint_positions'][name].append(position)

    def monitor_performance(self):
        """Monitor system performance"""
        # Simulate CPU and memory monitoring
        # In a real implementation, you would use psutil or similar
        import random
        cpu_usage = random.uniform(10, 80)  # Simulated CPU usage
        memory_usage = random.uniform(20, 70)  # Simulated memory usage

        self.metrics['cpu_usage'].append(cpu_usage)
        self.metrics['memory_usage'].append(memory_usage)

        # Publish current usage
        cpu_msg = Float32()
        cpu_msg.data = cpu_usage
        self.cpu_usage_pub.publish(cpu_msg)

        memory_msg = Float32()
        memory_msg.data = memory_usage
        self.memory_usage_pub.publish(memory_msg)

    def report_performance(self):
        """Report performance metrics"""
        if not self.metrics['control_loop_times']:
            return

        # Calculate statistics
        control_times = list(self.metrics['control_loop_times'])
        cpu_usage = list(self.metrics['cpu_usage'])
        memory_usage = list(self.metrics['memory_usage'])

        performance_data = {
            'timestamp': self.get_clock().now().nanoseconds / 1e9,
            'control_loop': {
                'mean_time': statistics.mean(control_times),
                'std_dev': statistics.stdev(control_times) if len(control_times) > 1 else 0,
                'min_time': min(control_times),
                'max_time': max(control_times),
                'rate_hz': 1.0 / statistics.mean(control_times) if statistics.mean(control_times) > 0 else 0
            },
            'system_resources': {
                'cpu_avg': statistics.mean(cpu_usage) if cpu_usage else 0,
                'cpu_peak': max(cpu_usage) if cpu_usage else 0,
                'memory_avg': statistics.mean(memory_usage) if memory_usage else 0,
                'memory_peak': max(memory_usage) if memory_usage else 0
            },
            'joint_metrics': {}
        }

        # Add joint-specific metrics
        for joint_name, positions in self.metrics['joint_positions'].items():
            if len(positions) > 1:
                pos_list = list(positions)
                performance_data['joint_metrics'][joint_name] = {
                    'mean_position': statistics.mean(pos_list),
                    'position_variance': statistics.variance(pos_list) if len(pos_list) > 1 else 0,
                    'range': max(pos_list) - min(pos_list)
                }

        # Publish performance data
        perf_msg = String()
        perf_msg.data = json.dumps(performance_data, indent=2)
        self.performance_pub.publish(perf_msg)

        # Log warnings if performance degrades
        mean_loop_time = performance_data['control_loop']['mean_time']
        if mean_loop_time > 0.05:  # 50ms threshold
            self.get_logger().warn(f'Control loop time degraded: {mean_loop_time:.3f}s')

        cpu_avg = performance_data['system_resources']['cpu_avg']
        if cpu_avg > 80:  # 80% CPU threshold
            self.get_logger().warn(f'High CPU usage detected: {cpu_avg:.1f}%')

def main(args=None):
    rclpy.init(args=args)
    monitor = PerformanceMonitor()

    try:
        rclpy.spin(monitor)
    except KeyboardInterrupt:
        monitor.get_logger().info('Shutting down performance monitor')
    finally:
        monitor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 6. Interactive Debugging Tools

### Interactive Debugging Interface
```python
# interactive_debugger.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
import json
import threading
import time

class InteractiveDebugger(Node):
    def __init__(self):
        super().__init__('interactive_debugger')

        # Debug state
        self.debug_mode = False
        self.paused = False
        self.breakpoints = []
        self.watch_variables = {}
        self.debug_history = []

        # Publishers and subscribers
        self.debug_command_pub = self.create_publisher(String, '/debug_commands', 10)
        self.debug_status_pub = self.create_publisher(String, '/debug_status', 10)

        # Robot state subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )

        # Timer for debug monitoring
        self.debug_timer = self.create_timer(0.1, self.debug_monitor)

        # Command subscriber
        self.debug_cmd_sub = self.create_subscription(
            String, '/debug_user_commands', self.debug_command_callback, 10
        )

        self.get_logger().info('Interactive Debugger initialized')

    def joint_state_callback(self, msg):
        """Monitor joint states for debugging"""
        if self.debug_mode:
            self.watch_variables['joint_states'] = {
                'names': msg.name,
                'positions': list(msg.position),
                'velocities': list(msg.velocity),
                'efforts': list(msg.effort),
                'timestamp': self.get_clock().now().nanoseconds / 1e9
            }

    def imu_callback(self, msg):
        """Monitor IMU data for debugging"""
        if self.debug_mode:
            self.watch_variables['imu'] = {
                'orientation': [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
                'angular_velocity': [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
                'linear_acceleration': [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z],
                'timestamp': self.get_clock().now().nanoseconds / 1e9
            }

    def debug_command_callback(self, msg):
        """Handle user debug commands"""
        try:
            command_data = json.loads(msg.data)
            command = command_data.get('command', '')
            params = command_data.get('params', {})

            if command == 'toggle_debug':
                self.debug_mode = not self.debug_mode
                self.get_logger().info(f'Debug mode: {self.debug_mode}')
            elif command == 'pause':
                self.paused = True
                self.get_logger().info('Execution paused')
            elif command == 'resume':
                self.paused = False
                self.get_logger().info('Execution resumed')
            elif command == 'add_breakpoint':
                self.add_breakpoint(params)
            elif command == 'remove_breakpoint':
                self.remove_breakpoint(params)
            elif command == 'step':
                self.step_execution()
            elif command == 'watch':
                self.add_watch_variable(params)
            elif command == 'get_state':
                self.publish_debug_state()

        except json.JSONDecodeError:
            self.get_logger().error(f'Invalid debug command: {msg.data}')

    def add_breakpoint(self, params):
        """Add a breakpoint for debugging"""
        condition = params.get('condition', '')
        if condition and condition not in self.breakpoints:
            self.breakpoints.append(condition)
            self.get_logger().info(f'Added breakpoint: {condition}')

    def remove_breakpoint(self, params):
        """Remove a breakpoint"""
        condition = params.get('condition', '')
        if condition in self.breakpoints:
            self.breakpoints.remove(condition)
            self.get_logger().info(f'Removed breakpoint: {condition}')

    def step_execution(self):
        """Step execution for debugging"""
        if self.paused:
            self.paused = False
            self.get_logger().info('Stepped execution')

    def add_watch_variable(self, params):
        """Add a variable to watch during debugging"""
        var_name = params.get('name', '')
        var_path = params.get('path', '')
        if var_name and var_path:
            self.watch_variables[var_name] = {'path': var_path, 'value': None}
            self.get_logger().info(f'Watching variable: {var_name}')

    def debug_monitor(self):
        """Monitor for debug conditions"""
        if self.debug_mode and not self.paused:
            # Check breakpoints
            for condition in self.breakpoints:
                if self.evaluate_condition(condition):
                    self.paused = True
                    self.get_logger().info(f'Breakpoint hit: {condition}')
                    break

            # Update watch variables
            self.update_watch_variables()

        # Publish debug status
        status_msg = String()
        status_data = {
            'debug_mode': self.debug_mode,
            'paused': self.paused,
            'breakpoints': self.breakpoints,
            'watch_variables': self.watch_variables
        }
        status_msg.data = json.dumps(status_data)
        self.debug_status_pub.publish(status_msg)

    def evaluate_condition(self, condition):
        """Evaluate a debug condition"""
        # This would evaluate actual robot state conditions
        # For now, return False to avoid breaking execution
        return False

    def update_watch_variables(self):
        """Update watched variables"""
        # Update with current robot state
        pass

    def publish_debug_state(self):
        """Publish current debug state"""
        debug_state = {
            'debug_mode': self.debug_mode,
            'paused': self.paused,
            'breakpoints': self.breakpoints,
            'watch_variables': self.watch_variables,
            'robot_state': self.watch_variables
        }

        state_msg = String()
        state_msg.data = json.dumps(debug_state)
        self.debug_status_pub.publish(state_msg)

def main(args=None):
    rclpy.init(args=args)
    debugger = InteractiveDebugger()

    try:
        rclpy.spin(debugger)
    except KeyboardInterrupt:
        debugger.get_logger().info('Shutting down interactive debugger')
    finally:
        debugger.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 7. Test Data Analysis and Visualization

### Test Data Analysis Node
```python
# test_data_analyzer.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
import json
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime
import os

class TestDataAnalyzer(Node):
    def __init__(self):
        super().__init__('test_data_analyzer')

        # Data storage
        self.test_data = defaultdict(list)
        self.test_sessions = {}
        self.current_session_id = None

        # Subscribers for test data
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )
        self.test_result_sub = self.create_subscription(
            String, '/test_results', self.test_result_callback, 10
        )

        # Command subscriber
        self.analysis_cmd_sub = self.create_subscription(
            String, '/analysis_commands', self.analysis_command_callback, 10
        )

        # Initialize test session
        self.start_new_session()

        self.get_logger().info('Test Data Analyzer initialized')

    def start_new_session(self):
        """Start a new test data collection session"""
        self.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.test_sessions[self.current_session_id] = {
            'start_time': datetime.now(),
            'data_points': 0,
            'tests_run': []
        }
        self.get_logger().info(f'Started new test session: {self.current_session_id}')

    def joint_state_callback(self, msg):
        """Collect joint state data"""
        timestamp = self.get_clock().now().nanoseconds / 1e9
        joint_data = {
            'timestamp': timestamp,
            'names': msg.name,
            'positions': list(msg.position),
            'velocities': list(msg.velocity),
            'efforts': list(msg.effort)
        }
        self.test_data['joint_states'].append(joint_data)

        session = self.test_sessions[self.current_session_id]
        session['data_points'] += 1

    def imu_callback(self, msg):
        """Collect IMU data"""
        timestamp = self.get_clock().now().nanoseconds / 1e9
        imu_data = {
            'timestamp': timestamp,
            'orientation': [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
            'angular_velocity': [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
            'linear_acceleration': [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
        }
        self.test_data['imu'].append(imu_data)

    def test_result_callback(self, msg):
        """Collect test results"""
        try:
            result = json.loads(msg.data)
            result['timestamp'] = self.get_clock().now().nanoseconds / 1e9
            self.test_data['test_results'].append(result)

            session = self.test_sessions[self.current_session_id]
            session['tests_run'].append(result)

        except json.JSONDecodeError:
            self.get_logger().error(f'Invalid test result format: {msg.data}')

    def analysis_command_callback(self, msg):
        """Handle analysis commands"""
        try:
            command_data = json.loads(msg.data)
            command = command_data.get('command', '')
            params = command_data.get('params', {})

            if command == 'generate_report':
                self.generate_test_report()
            elif command == 'plot_joint_trajectories':
                self.plot_joint_trajectories()
            elif command == 'plot_balance_metrics':
                self.plot_balance_metrics()
            elif command == 'export_data':
                self.export_test_data(params.get('format', 'json'))
            elif command == 'compare_sessions':
                self.compare_sessions(params.get('session_ids', []))

        except json.JSONDecodeError:
            self.get_logger().error(f'Invalid analysis command: {msg.data}')

    def generate_test_report(self):
        """Generate comprehensive test report"""
        report = {
            'session_id': self.current_session_id,
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_data_points': len(self.test_data['joint_states']) + len(self.test_data['imu']),
                'tests_completed': len(self.test_data['test_results']),
                'session_duration': (datetime.now() - self.test_sessions[self.current_session_id]['start_time']).total_seconds()
            },
            'joint_analysis': self.analyze_joint_data(),
            'balance_analysis': self.analyze_balance_data(),
            'performance_summary': self.analyze_performance()
        }

        # Save report
        report_filename = f"test_report_{self.current_session_id}.json"
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        self.get_logger().info(f'Test report generated: {report_filename}')

    def analyze_joint_data(self):
        """Analyze joint state data"""
        if not self.test_data['joint_states']:
            return {}

        # Extract joint positions over time
        joint_names = set()
        for data in self.test_data['joint_states']:
            joint_names.update(data['names'])

        analysis = {}
        for joint in joint_names:
            positions = []
            timestamps = []

            for data in self.test_data['joint_states']:
                if joint in data['names']:
                    idx = data['names'].index(joint)
                    positions.append(data['positions'][idx])
                    timestamps.append(data['timestamp'])

            if positions:
                analysis[joint] = {
                    'mean_position': float(np.mean(positions)),
                    'std_position': float(np.std(positions)),
                    'min_position': float(np.min(positions)),
                    'max_position': float(np.max(positions)),
                    'range': float(np.max(positions) - np.min(positions)),
                    'data_points': len(positions)
                }

        return analysis

    def analyze_balance_data(self):
        """Analyze balance and stability metrics from IMU data"""
        if not self.test_data['imu']:
            return {}

        orientations = []
        accelerations = []

        for data in self.test_data['imu']:
            # Convert quaternion to roll/pitch/yaw for balance analysis
            quat = data['orientation']
            roll, pitch, yaw = self.quaternion_to_euler(quat)
            orientations.extend([roll, pitch, yaw])

            # Linear acceleration magnitude
            acc = data['linear_acceleration']
            acc_mag = np.sqrt(acc[0]**2 + acc[1]**2 + acc[2]**2)
            accelerations.append(acc_mag)

        if accelerations:
            return {
                'avg_orientation': float(np.mean(np.abs(orientations))),
                'max_orientation_deviation': float(np.max(np.abs(orientations))),
                'avg_acceleration': float(np.mean(accelerations)),
                'max_acceleration': float(np.max(accelerations)),
                'stability_score': 1.0 - min(1.0, float(np.std(accelerations)))  # Lower std = more stable
            }

        return {}

    def analyze_performance(self):
        """Analyze overall performance metrics"""
        if not self.test_data['test_results']:
            return {}

        successful_tests = [t for t in self.test_data['test_results'] if t.get('success', False)]
        success_rate = len(successful_tests) / len(self.test_data['test_results']) if self.test_data['test_results'] else 0

        return {
            'success_rate': success_rate,
            'total_tests': len(self.test_data['test_results']),
            'successful_tests': len(successful_tests),
            'failed_tests': len(self.test_data['test_results']) - len(successful_tests)
        }

    def plot_joint_trajectories(self):
        """Plot joint position trajectories"""
        if not self.test_data['joint_states']:
            self.get_logger().warn('No joint state data to plot')
            return

        # Group data by joint
        joint_data = defaultdict(lambda: {'time': [], 'pos': []})
        for data in self.test_data['joint_states']:
            for name, pos in zip(data['names'], data['positions']):
                joint_data[name]['time'].append(data['timestamp'])
                joint_data[name]['pos'].append(pos)

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        for joint_name, data in joint_data.items():
            ax.plot(data['time'], data['pos'], label=joint_name, alpha=0.7)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Joint Position (rad)')
        ax.set_title('Joint Position Trajectories')
        ax.legend()
        ax.grid(True)

        # Save plot
        plot_filename = f"joint_trajectories_{self.current_session_id}.png"
        plt.savefig(plot_filename)
        plt.close()
        self.get_logger().info(f'Joint trajectories plot saved: {plot_filename}')

    def plot_balance_metrics(self):
        """Plot balance and stability metrics"""
        if not self.test_data['imu']:
            self.get_logger().warn('No IMU data to plot')
            return

        # Extract orientation data
        times = []
        rolls = []
        pitches = []
        yaws = []

        for data in self.test_data['imu']:
            quat = data['orientation']
            roll, pitch, yaw = self.quaternion_to_euler(quat)
            times.append(data['timestamp'])
            rolls.append(roll)
            pitches.append(pitch)
            yaws.append(yaw)

        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot orientations
        ax1.plot(times, rolls, label='Roll', alpha=0.7)
        ax1.plot(times, pitches, label='Pitch', alpha=0.7)
        ax1.plot(times, yaws, label='Yaw', alpha=0.7)
        ax1.set_ylabel('Angle (rad)')
        ax1.set_title('Robot Orientation Over Time')
        ax1.legend()
        ax1.grid(True)

        # Plot acceleration
        accelerations = []
        acc_times = []
        for data in self.test_data['imu']:
            acc = data['linear_acceleration']
            acc_mag = np.sqrt(acc[0]**2 + acc[1]**2 + acc[2]**2)
            accelerations.append(acc_mag)
            acc_times.append(data['timestamp'])

        ax2.plot(acc_times, accelerations, 'r-', label='Acceleration Magnitude', alpha=0.7)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Acceleration (m/s²)')
        ax2.set_title('Linear Acceleration Magnitude')
        ax2.legend()
        ax2.grid(True)

        # Save plot
        plot_filename = f"balance_metrics_{self.current_session_id}.png"
        plt.savefig(plot_filename)
        plt.close()
        self.get_logger().info(f'Balance metrics plot saved: {plot_filename}')

    def quaternion_to_euler(self, quat):
        """Convert quaternion to Euler angles"""
        x, y, z, w = quat

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def export_test_data(self, format='json'):
        """Export test data in specified format"""
        export_data = {
            'session_id': self.current_session_id,
            'export_time': datetime.now().isoformat(),
            'test_data': dict(self.test_data)
        }

        if format == 'json':
            filename = f"test_data_{self.current_session_id}.json"
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        elif format == 'csv':
            # Export to CSV format
            self.export_to_csv(export_data)

        self.get_logger().info(f'Test data exported: {filename}')

    def export_to_csv(self, export_data):
        """Export data to CSV format"""
        # Convert to pandas DataFrame and save as CSV
        pass

def main(args=None):
    rclpy.init(args=args)
    analyzer = TestDataAnalyzer()

    try:
        rclpy.spin(analyzer)
    except KeyboardInterrupt:
        analyzer.get_logger().info('Shutting down test data analyzer')
    finally:
        analyzer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 8. Best Practices for Interactive Testing

### Testing Guidelines and Checklists:

1. **Pre-Testing Checklist**:
   - Verify simulation environment is properly configured
   - Check that all sensors are publishing data
   - Ensure robot model is properly loaded and positioned
   - Confirm ROS 2 communication is working correctly

2. **During Testing**:
   - Monitor robot state continuously
   - Watch for unexpected behaviors or errors
   - Keep detailed notes of observations
   - Test incrementally, starting with simple behaviors

3. **Post-Testing**:
   - Analyze all collected data
   - Document findings and issues
   - Compare results with expected outcomes
   - Plan improvements for next iteration

### Performance Optimization:
- Use appropriate update rates for different types of monitoring
- Implement data buffering to avoid overwhelming the system
- Use efficient data structures for real-time analysis
- Profile tools to identify performance bottlenecks

## 9. Troubleshooting Interactive Testing Issues

### Common Issues and Solutions:

1. **High CPU Usage During Testing**:
   - Reduce visualization update rates
   - Use Level of Detail (LOD) for complex models
   - Limit the number of concurrent visualizations
   - Optimize data collection frequency

2. **Timing Issues**:
   - Ensure proper time synchronization between nodes
   - Use simulation time when appropriate
   - Check for blocking operations in callbacks
   - Monitor real-time factor in simulation

3. **Data Corruption or Loss**:
   - Use appropriate QoS settings for reliable communication
   - Implement proper error checking and recovery
   - Use message buffering when needed
   - Verify network connectivity for distributed testing

4. **Visualization Problems**:
   - Check coordinate frame transformations
   - Verify proper scaling between simulation and visualization
   - Ensure visualization tools are compatible with data formats
   - Monitor graphics performance and adjust settings accordingly

## 10. Integration with CI/CD and Automated Testing

### Continuous Integration Testing:
```bash
# Example CI/CD pipeline for testing
# .github/workflows/test_simulation.yml
name: Simulation Testing
on: [push, pull_request]

jobs:
  simulation-test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2

    - name: Setup ROS 2
      uses: ros-tooling/setup-ros@0.7.3
      with:
        required-ros-distributions: humble

    - name: Install dependencies
      run: |
        sudo apt update
        sudo apt install -y ros-humble-gazebo-ros-pkgs ros-humble-robot-state-publisher

    - name: Build workspace
      run: |
        colcon build --packages-select humanoid_simulation

    - name: Run tests
      run: |
        source install/setup.bash
        ros2 launch humanoid_simulation test_scenario.launch.py
```

## Weekly Schedule Focus (Weeks 6-7)
During Weeks 6-7, we will focus on:
- Creating interactive testing environments
- Implementing debugging and visualization tools
- Developing comprehensive test scenarios
- Validating robot behavior in simulation

## Resources
- [ROS 2 Testing Guidelines](https://docs.ros.org/en/humble/The-ROS2-Project/Contributing/Code-Style-Language-Versions.html#testing)
- [Gazebo Testing Tools](http://gazebosim.org/tutorials?tut=ros_gzplugins#Testing-tools)
- [Robot Testing Framework](https://github.com/ros-planning/robot-configuration-testing)
- [Simulation Best Practices](https://github.com/ros-simulation/gazebo_ros_pkgs/wiki)
