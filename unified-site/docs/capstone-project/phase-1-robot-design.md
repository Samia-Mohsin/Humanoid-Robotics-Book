# Phase 1: Robot Design

## Overview

Phase 1 focuses on designing and configuring the humanoid robot in simulation. This phase establishes the foundation for all subsequent phases by creating a properly modeled robot with appropriate kinematics, dynamics, and sensor configurations that will enable the implementation of locomotion, perception, and action capabilities.

## Learning Objectives

By the end of this phase, you will be able to:

- Design a humanoid robot using URDF (Unified Robot Description Format)
- Configure appropriate sensors for perception and navigation
- Establish ROS 2 communication framework for the robot
- Validate the robot model through simulation testing

## Weekly Breakdown

### Week 1: Humanoid Robot Architecture Design

**Learning Goals:**
- Understand the kinematic structure of humanoid robots
- Design joint configurations for stable locomotion
- Plan sensor placement for effective perception

**Activities:**
- Study existing humanoid robot architectures (Atlas, ASIMO, Pepper, etc.)
- Design kinematic chain for legs, arms, and torso
- Determine degrees of freedom for each body part
- Plan center of mass considerations for balance

**Deliverables:**
- Robot kinematic diagram
- Joint configuration specification
- Initial URDF skeleton

### Week 2: URDF Modeling and Simulation Setup

**Learning Goals:**
- Create detailed URDF models with accurate physical properties
- Configure Gazebo plugins for realistic simulation
- Set up ROS 2 interfaces for robot control

**Activities:**
- Develop complete URDF with links, joints, and inertial properties
- Configure Gazebo plugins for physics simulation
- Set up ROS 2 control interfaces (joint_state_publisher, robot_state_publisher)
- Validate kinematic model in RViz and Gazebo

**Deliverables:**
- Complete URDF model
- Gazebo world file with robot spawn
- ROS 2 launch files for robot simulation

## Robot Design Requirements

### Physical Specifications

Your humanoid robot must meet the following specifications:

- **Height**: 1.2 - 1.5 meters (adjustable based on application)
- **Degrees of Freedom**: Minimum 24 DOF (6 per leg, 6 per arm, 2 for torso, 2 for head)
- **Actuators**: Simulated servo motors with appropriate torque specifications
- **Sensors**: RGB-D camera, IMU, force/torque sensors, joint position/velocity/torque feedback

### Kinematic Structure

The robot should include:

- **Torso**: Pitch and yaw joints for upper body movement
- **Head**: Pan and tilt joints for gaze control
- **Arms**: Shoulders (3 DOF), elbows (1 DOF), wrists (2 DOF) per arm
- **Legs**: Hips (3 DOF), knees (1 DOF), ankles (2 DOF) per leg

### Sensor Configuration

Critical sensors for the humanoid robot:

- **RGB-D Camera**: For vision-based perception and navigation
- **IMU**: For balance and orientation sensing
- **Force/Torque Sensors**: In feet and hands for contact detection
- **Joint Sensors**: Position, velocity, and effort feedback for all joints

## Implementation Steps

### Step 1: Create Base URDF Structure

Create the foundational URDF file with the robot's main components:

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot">
  <!-- Robot base link -->
  <link name="base_link">
    <inertial>
      <mass value="10.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.3 0.2 0.4"/>
      </geometry>
      <material name="grey">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.3 0.2 0.4"/>
      </geometry>
    </collision>
  </link>

  <!-- Additional links and joints would be defined here -->
</robot>
```

### Step 2: Add Joint Definitions

Define the joints connecting different body parts:

```xml
  <!-- Example joint definition -->
  <joint name="torso_joint" type="fixed">
    <parent link="base_link"/>
    <child link="torso_link"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
  </joint>

  <!-- Example of a revolute joint -->
  <joint name="hip_joint_r" type="revolute">
    <parent link="base_link"/>
    <child link="right_hip_link"/>
    <origin xyz="-0.1 -0.1 -0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0"/>
    <dynamics damping="1.0" friction="0.1"/>
  </joint>
```

### Step 3: Configure Gazebo Integration

Add Gazebo-specific configurations for physics simulation:

```xml
  <!-- Gazebo plugin for ROS control -->
  <gazebo>
    <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
      <robotNamespace>/humanoid_robot</robotNamespace>
    </plugin>
  </gazebo>

  <!-- Sensor configurations -->
  <gazebo reference="camera_link">
    <sensor type="depth" name="camera">
      <always_on>true</always_on>
      <visualize>true</visualize>
      <camera>
        <horizontal_fov>1.047</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.1</near>
          <far>10.0</far>
        </clip>
      </camera>
    </sensor>
  </gazebo>
```

### Step 4: Create ROS 2 Launch Files

Create launch files to spawn and control the robot:

```python
# launch/humanoid_robot.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.substitutions import Command
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get the urdf file location
    urdf_dir = get_package_share_directory('humanoid_robot_description')
    urdf_file = os.path.join(urdf_dir, 'urdf', 'humanoid_robot.urdf.xacro')

    # Robot state publisher node
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{'robot_description': Command(['xacro ', urdf_file])}]
    )

    # Joint state publisher node
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher'
    )

    return LaunchDescription([
        robot_state_publisher,
        joint_state_publisher
    ])
```

## Validation and Testing

### Kinematic Validation

- Verify that the robot's kinematic chain is properly defined
- Test range of motion for all joints
- Ensure no kinematic loops or inconsistencies

### Dynamic Validation

- Validate mass and inertia properties
- Test balance and stability in simulation
- Verify sensor data accuracy

### ROS 2 Interface Validation

- Confirm all topics are properly published
- Test TF tree completeness
- Verify control interfaces functionality

## Assessment Rubric

Your Phase 1 implementation will be evaluated on:

- **URDF Quality (30%)**: Properly structured URDF with correct physical properties
- **Sensor Integration (25%)**: Correct placement and configuration of all required sensors
- **Kinematic Design (25%)**: Appropriate joint configurations for humanoid movement
- **ROS 2 Integration (20%)**: Proper setup of ROS 2 communication framework

## Resources

- [ROS URDF Documentation](http://wiki.ros.org/urdf)
- [Gazebo Robot Modeling Tutorials](http://gazebosim.org/tutorials?tut=ros_urdf)
- [NVIDIA Isaac URDF Examples](https://docs.nvidia.com/isaac/isaac_sim/tutorials/create_urdf.html)
- [Humanoid Robot Design Guidelines](https://humanoids.wiki/Robot_Design)

## Next Phase

Upon successful completion of Phase 1, you will proceed to Phase 2: Locomotion, where you will implement bipedal walking and balance control systems for your humanoid robot.
