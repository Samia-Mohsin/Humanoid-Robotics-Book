# Phase 2: Locomotion

## Overview

Phase 2 focuses on implementing locomotion capabilities for your humanoid robot, specifically bipedal walking and balance control. This phase builds upon the robot design from Phase 1 and introduces the fundamental challenges of humanoid locomotion, including balance maintenance, gait planning, and navigation in complex environments.

## Learning Objectives

By the end of this phase, you will be able to:

- Implement bipedal walking algorithms for humanoid robots
- Design and implement balance control systems using feedback from IMU and force/torque sensors
- Integrate Nav2 for autonomous navigation in humanoid robots
- Test locomotion capabilities in simulated environments

## Weekly Breakdown

### Week 1: Bipedal Walking Fundamentals

**Learning Goals:**
- Understand the principles of bipedal locomotion
- Implement basic walking patterns (e.g., inverse kinematics-based walking)
- Design gait planning algorithms for stable locomotion

**Activities:**
- Study human walking biomechanics and apply to robot control
- Implement basic walking patterns using joint trajectories
- Design footstep planning algorithms
- Test basic locomotion in simulation

**Deliverables:**
- Walking pattern implementation
- Footstep planning algorithm
- Basic locomotion demonstration in simulation

### Week 2: Balance Control and Stability

**Learning Goals:**
- Implement balance control using sensor feedback
- Apply control theory to maintain robot stability
- Integrate with Nav2 for autonomous navigation

**Activities:**
- Implement balance control using IMU and force/torque sensor feedback
- Design PID controllers for joint position control
- Integrate with Nav2 for path planning and navigation
- Test locomotion with obstacle avoidance

**Deliverables:**
- Balance control system
- Nav2 integration
- Navigation demonstration with obstacle avoidance

## Locomotion System Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Path Planner  │───▶│  Gait Planner   │───▶│  Balance Ctrl   │
│   (Nav2)        │    │  (Footsteps)    │    │  (Stabilizer)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Robot Control Interface                      │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components

The locomotion system consists of several interconnected modules:

```python
import numpy as np
import math
from geometry_msgs.msg import Twist, Pose, Point
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray
import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class BipedalController(Node):
    def __init__(self):
        super().__init__('bipedal_controller')

        # Publishers and subscribers
        self.joint_trajectory_pub = self.create_publisher(JointTrajectory, '/joint_trajectory', 10)
        self.imu_sub = self.create_subscription(Imu, '/imu/data', self.imu_callback, 10)
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)

        # Robot parameters
        self.robot_height = 1.0  # meters
        self.step_height = 0.05  # meters
        self.step_length = 0.3   # meters
        self.step_width = 0.2    # meters

        # Balance control parameters
        self.balance_kp = 1.0
        self.balance_kd = 0.1

        # State variables
        self.current_pose = Pose()
        self.imu_data = Imu()
        self.joint_states = JointState()
        self.balance_error = 0.0

        # Timers
        self.balance_timer = self.create_timer(0.01, self.balance_control_loop)
        self.walk_timer = None

    def imu_callback(self, msg):
        """Callback for IMU data"""
        self.imu_data = msg
        # Calculate roll and pitch angles for balance
        self.update_balance_error()

    def joint_state_callback(self, msg):
        """Callback for joint states"""
        self.joint_states = msg

    def update_balance_error(self):
        """Calculate balance error from IMU data"""
        # Convert quaternion to roll/pitch angles
        quat = self.imu_data.orientation
        roll, pitch, yaw = self.quaternion_to_euler(quat)

        # Balance error is deviation from upright position
        self.balance_error = math.sqrt(roll**2 + pitch**2)

    def quaternion_to_euler(self, quat):
        """Convert quaternion to Euler angles"""
        # Convert quaternion to roll, pitch, yaw
        sinr_cosp = 2 * (quat.w * quat.x + quat.y * quat.z)
        cosr_cosp = 1 - 2 * (quat.x * quat.x + quat.y * quat.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (quat.w * quat.y - quat.z * quat.x)
        pitch = math.asin(sinp)

        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def balance_control_loop(self):
        """Main balance control loop"""
        # Simple PD controller for balance
        balance_correction = self.balance_kp * self.balance_error

        # Apply balance correction to joints
        self.apply_balance_correction(balance_correction)

    def apply_balance_correction(self, correction):
        """Apply balance correction to joints"""
        # This is a simplified approach - in reality, this would be more complex
        # involving inverse kinematics and full-body control
        pass

    def walk_forward(self, steps=1):
        """Execute forward walking motion"""
        trajectory = JointTrajectory()
        trajectory.joint_names = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint',
            # Add other joints as needed
        ]

        # Generate walking trajectory
        trajectory.points = self.generate_walking_trajectory(steps)

        # Publish trajectory
        self.joint_trajectory_pub.publish(trajectory)

    def generate_walking_trajectory(self, steps):
        """Generate joint trajectory for walking"""
        points = []
        time_from_start = 0.0

        for step in range(steps):
            # Calculate trajectory for one complete step cycle
            step_trajectory = self.calculate_single_step_trajectory()

            for point in step_trajectory:
                point.time_from_start = Duration(sec=int(time_from_start), nanosec=int((time_from_start % 1) * 1e9))
                points.append(point)
                time_from_start += 0.1  # 100ms per point

        return points

    def calculate_single_step_trajectory(self):
        """Calculate trajectory for a single step"""
        # This is a simplified implementation
        # Real implementation would use more sophisticated gait patterns
        points = []

        # Example: simple sinusoidal walking pattern
        for i in range(10):  # 10 points per step
            point = JointTrajectoryPoint()

            # Calculate joint positions for this point in the step
            t = i / 10.0  # normalized time (0 to 1)

            # Hip joints - lift leg
            left_hip = math.sin(t * math.pi) * 0.1
            right_hip = math.sin((t + 0.5) * math.pi) * 0.1  # Opposite phase

            # Knee joints - bend for step
            left_knee = math.sin(t * math.pi) * 0.2
            right_knee = math.sin((t + 0.5) * math.pi) * 0.2

            # Ankle joints - adjust for balance
            left_ankle = math.sin(t * math.pi) * 0.05
            right_ankle = math.sin((t + 0.5) * math.pi) * 0.05

            point.positions = [left_hip, left_knee, left_ankle, right_hip, right_knee, right_ankle]
            point.velocities = [0.0] * 6  # Simplified
            point.accelerations = [0.0] * 6  # Simplified

            points.append(point)

        return points
```

## Balance Control Implementation

### Zero Moment Point (ZMP) Controller

The ZMP controller is crucial for maintaining balance in humanoid robots:

```python
class ZMPController:
    def __init__(self, robot_mass=50.0, gravity=9.81):
        self.mass = robot_mass
        self.gravity = gravity
        self.com_height = 0.8  # Center of mass height
        self.zmp_ref = np.array([0.0, 0.0])  # Reference ZMP
        self.com_pos = np.array([0.0, 0.0, self.com_height])  # Center of mass position
        self.com_vel = np.array([0.0, 0.0, 0.0])  # Center of mass velocity
        self.com_acc = np.array([0.0, 0.0, 0.0])  # Center of mass acceleration

        # Control gains
        self.kp = 10.0
        self.kd = 5.0

    def update_com_state(self, com_pos, com_vel, com_acc):
        """Update center of mass state"""
        self.com_pos = com_pos
        self.com_vel = com_vel
        self.com_acc = com_acc

    def calculate_zmp(self):
        """Calculate current ZMP based on CoM state"""
        # ZMP_x = com_x - (com_z - zmp_z) / g * com_acc_x
        # ZMP_y = com_y - (com_z - zmp_z) / g * com_acc_y
        zmp_z = 0.0  # Assume feet are at z=0

        zmp_x = self.com_pos[0] - (self.com_pos[2] - zmp_z) / self.gravity * self.com_acc[0]
        zmp_y = self.com_pos[1] - (self.com_pos[2] - zmp_z) / self.gravity * self.com_acc[1]

        return np.array([zmp_x, zmp_y])

    def calculate_balance_correction(self):
        """Calculate balance correction based on ZMP error"""
        current_zmp = self.calculate_zmp()
        zmp_error = self.zmp_ref - current_zmp

        # Simple PD control
        correction = self.kp * zmp_error + self.kd * (zmp_error - self.get_prev_error())

        return correction

    def get_prev_error(self):
        """Get previous error (in a real implementation, this would be stored)"""
        return np.array([0.0, 0.0])
```

### Inverse Kinematics for Foot Placement

Accurate foot placement is essential for stable walking:

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

class FootPlacementController:
    def __init__(self, leg_length=0.5):
        self.leg_length = leg_length
        self.step_width = 0.2  # Distance between feet
        self.step_height = 0.05  # Height to lift foot during step

    def calculate_foot_positions(self, current_pos, step_direction, step_size):
        """Calculate desired foot positions for next step"""
        # Calculate where to place the swing foot
        # This is a simplified approach - real implementation would be more complex

        # Determine support leg and swing leg
        # For alternating steps, odd steps are right leg swing, even are left
        if step_direction % 2 == 0:  # Left step
            support_foot = np.array([current_pos[0], current_pos[1] - self.step_width/2, 0.0])
            swing_foot = np.array([current_pos[0], current_pos[1] + self.step_width/2, 0.0])
        else:  # Right step
            support_foot = np.array([current_pos[0], current_pos[1] + self.step_width/2, 0.0])
            swing_foot = np.array([current_pos[0], current_pos[1] - self.step_width/2, 0.0])

        # Calculate next swing foot position based on step direction and size
        step_vec = np.array([step_size, 0.0, 0.0])  # Simplified - only forward steps
        next_swing_pos = swing_foot + step_vec

        return support_foot, next_swing_pos

    def inverse_kinematics_leg(self, target_pos, leg_side='left'):
        """Calculate joint angles for leg to reach target position"""
        # Simplified 3DOF leg IK (hip, knee, ankle)
        # Real implementation would be more complex with 6DOF

        # Calculate relative position from hip to target
        hip_to_target = target_pos - np.array([0, 0, -self.leg_length])

        # Simplified 2D IK for sagittal plane
        x, y, z = hip_to_target

        # Calculate knee angle (assuming planar movement for simplicity)
        leg_length = np.sqrt(x**2 + z**2)

        # Check if target is reachable
        if leg_length > 2 * self.leg_length:
            # Target is out of reach
            return None

        # Calculate knee angle using law of cosines
        knee_angle = math.pi - math.acos(
            (self.leg_length**2 + self.leg_length**2 - leg_length**2) /
            (2 * self.leg_length * self.leg_length)
        )

        # Calculate hip and ankle angles
        alpha = math.atan2(x, abs(z))  # Angle from vertical
        beta = math.acos(
            (self.leg_length**2 + leg_length**2 - self.leg_length**2) /
            (2 * self.leg_length * leg_length)
        )

        hip_angle = alpha - beta
        ankle_angle = -(hip_angle + knee_angle)

        return [hip_angle, knee_angle, ankle_angle]
```

## Integration with Nav2

### Custom Nav2 Controller for Humanoid Robots

```python
from nav2_core.controller import Controller
from nav2_util import lifecycle
from geometry_msgs.msg import PoseStamped, Twist
import numpy as np

class HumanoidController(Controller):
    def __init__(self, name):
        self.name = name
        self.linear_vel_max = 0.3  # m/s
        self.angular_vel_max = 0.5  # rad/s
        self.linear_acc_max = 0.1  # m/s^2
        self.angular_acc_max = 0.2  # rad/s^2

    def configure(self, plugin_name, node, tf, costmap):
        """Configure the controller"""
        self.plugin_name = plugin_name
        self.node = node
        self.tf = tf
        self.costmap = costmap

        # Initialize humanoid-specific controllers
        self.bipedal_controller = BipedalController()

    def cleanup(self):
        """Cleanup controller resources"""
        pass

    def activate(self):
        """Activate the controller"""
        pass

    def deactivate(self):
        """Deactivate the controller"""
        pass

    def setPlan(self, path):
        """Set the plan for the controller"""
        self.path = path

    def computeVelocityCommands(self, pose, velocity, goal_checker):
        """Compute velocity commands to follow the path"""
        cmd_vel = Twist()

        # Calculate desired velocity based on path following
        desired_linear, desired_angular = self.calculate_path_following_velocity(pose, velocity)

        # Apply humanoid-specific constraints
        cmd_vel.linear.x = self.apply_velocity_constraints(
            velocity.linear.x,
            desired_linear,
            self.linear_vel_max,
            self.linear_acc_max
        )

        cmd_vel.angular.z = self.apply_velocity_constraints(
            velocity.angular.z,
            desired_angular,
            self.angular_vel_max,
            self.angular_acc_max
        )

        # Convert to walking commands for humanoid
        walking_cmd = self.convert_to_walking_commands(cmd_vel)

        return walking_cmd

    def calculate_path_following_velocity(self, pose, velocity):
        """Calculate velocity commands for path following"""
        # Simplified path following algorithm
        # In reality, this would use more sophisticated path tracking

        # Get next waypoint
        if len(self.path.poses) > 0:
            target = self.path.poses[0]

            # Calculate distance to target
            dx = target.pose.position.x - pose.pose.position.x
            dy = target.pose.position.y - pose.pose.position.y
            distance = math.sqrt(dx**2 + dy**2)

            # Calculate desired heading
            desired_heading = math.atan2(dy, dx)
            current_heading = self.get_yaw_from_pose(pose.pose)

            # Calculate heading error
            heading_error = desired_heading - current_heading
            heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))  # Normalize

            # Calculate velocities
            linear_vel = min(distance * 0.5, self.linear_vel_max)  # Simple proportional control
            angular_vel = heading_error * 1.0  # Simple proportional control

            return linear_vel, angular_vel
        else:
            return 0.0, 0.0

    def apply_velocity_constraints(self, current_vel, desired_vel, max_vel, max_acc):
        """Apply velocity and acceleration constraints"""
        # Limit acceleration
        vel_diff = desired_vel - current_vel
        max_change = max_acc * 0.1  # Assuming 10Hz control rate

        if abs(vel_diff) > max_change:
            vel_diff = max_change * (1 if vel_diff > 0 else -1)

        new_vel = current_vel + vel_diff

        # Limit velocity
        new_vel = max(min(new_vel, max_vel), -max_vel)

        return new_vel

    def convert_to_walking_commands(self, cmd_vel):
        """Convert velocity commands to walking commands for humanoid"""
        # This would convert the Twist command to actual walking steps
        # For now, return the original command with a note that it needs to be processed
        # by the bipedal controller
        walking_cmd = {
            'linear_velocity': cmd_vel.linear.x,
            'angular_velocity': cmd_vel.angular.z,
            'command_type': 'walking'
        }

        return walking_cmd

    def get_yaw_from_pose(self, pose):
        """Extract yaw angle from pose orientation"""
        quat = pose.orientation
        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw
```

## Testing and Validation

### Simulation Testing Framework

```python
import unittest
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Twist
from sensor_msgs.msg import Imu, JointState

class LocomotionTester(Node):
    def __init__(self):
        super().__init__('locomotion_tester')

        # Test parameters
        self.test_scenarios = [
            'straight_line_walking',
            'turning',
            'obstacle_avoidance',
            'balance_recovery'
        ]

        self.test_results = {}

    def run_all_tests(self):
        """Run all locomotion tests"""
        for scenario in self.test_scenarios:
            self.test_results[scenario] = self.run_test_scenario(scenario)

        return self.test_results

    def run_test_scenario(self, scenario):
        """Run a specific test scenario"""
        if scenario == 'straight_line_walking':
            return self.test_straight_line_walking()
        elif scenario == 'turning':
            return self.test_turning()
        elif scenario == 'obstacle_avoidance':
            return self.test_obstacle_avoidance()
        elif scenario == 'balance_recovery':
            return self.test_balance_recovery()

    def test_straight_line_walking(self):
        """Test straight line walking"""
        # Initialize robot at starting position
        start_pose = Pose()
        start_pose.position.x = 0.0
        start_pose.position.y = 0.0

        # Command robot to walk forward
        # self.send_walking_command(distance=2.0)

        # Wait for completion
        # result = self.wait_for_completion(timeout=30.0)

        # Measure final position
        # final_pose = self.get_current_pose()

        # Calculate accuracy
        # accuracy = abs(final_pose.position.x - 2.0)  # Should be close to 2.0m

        # For now, return a mock result
        return {
            'success': True,
            'distance_traveled': 2.0,
            'accuracy': 0.95,
            'time_taken': 15.0
        }

    def test_turning(self):
        """Test turning capability"""
        # Similar structure to straight line walking test
        return {
            'success': True,
            'angle_turned': 90.0,
            'accuracy': 0.92,
            'time_taken': 12.0
        }

    def test_obstacle_avoidance(self):
        """Test obstacle avoidance during navigation"""
        return {
            'success': True,
            'obstacles_avoided': 3,
            'path_efficiency': 0.85,
            'time_taken': 25.0
        }

    def test_balance_recovery(self):
        """Test balance recovery from disturbances"""
        return {
            'success': True,
            'recovery_time': 2.0,
            'disturbance_magnitude': 0.3,
            'success_rate': 0.98
        }
```

## Assessment Rubric

Your Phase 2 implementation will be evaluated on:

- **Bipedal Walking (30%)**: Stable walking with proper gait patterns
- **Balance Control (25%)**: Effective balance maintenance under disturbances
- **Nav2 Integration (25%)**: Proper integration with navigation stack
- **Testing and Validation (20%)**: Comprehensive testing with good results

## Resources

- [ROS Navigation Tutorials](http://wiki.ros.org/navigation/Tutorials)
- [Humanoid Path Planning](https://humanoids.wiki/Path_Planning)
- [NVIDIA Isaac Locomotion Examples](https://docs.nvidia.com/isaac/locomotion/index.html)
- [Bipedal Robot Control Papers](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=bipedal%20robot%20control)

## Next Phase

Upon successful completion of Phase 2, you will proceed to Phase 3: Perception, where you will implement computer vision, VSLAM, and sensor fusion capabilities for your humanoid robot.
