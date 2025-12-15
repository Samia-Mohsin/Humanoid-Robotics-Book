# Chapter 4: Understanding URDF (Unified Robot Description Format) for Humanoids

## Overview
This chapter provides an in-depth exploration of URDF (Unified Robot Description Format), the XML-based format used to describe robot models in ROS. For humanoid robots, URDF is essential for defining the complex kinematic structure, including multiple degrees of freedom and articulated joints that enable human-like movement.

## Learning Objectives
By the end of this chapter, students will be able to:
- Create detailed URDF models for humanoid robots
- Define complex joint structures and kinematic chains
- Specify visual, collision, and inertial properties
- Validate URDF models and troubleshoot common issues
- Integrate URDF with ROS 2 simulation and control systems

## 1. Introduction to URDF

URDF (Unified Robot Description Format) is an XML format used in ROS to describe robot models. For humanoid robots, URDF becomes particularly complex due to the multiple joints and links required to represent human-like structures with arms, legs, torso, and head.

### Key Components of URDF:
- **Links**: Rigid bodies that make up the robot
- **Joints**: Connections between links with specific degrees of freedom
- **Visual**: How the robot appears in simulation
- **Collision**: How the robot interacts with the environment
- **Inertial**: Physical properties for physics simulation

## 2. URDF Structure for Humanoid Robots

### Basic Humanoid URDF Template
```xml
<?xml version="1.0"?>
<robot name="humanoid_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Base/Fixed link -->
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
    </collision>
  </link>

  <!-- Torso -->
  <link name="torso">
    <inertial>
      <mass value="5.0"/>
      <origin xyz="0 0 0.3" rpy="0 0 0"/>
      <inertia ixx="0.5" ixy="0" ixz="0" iyy="0.5" iyz="0" izz="0.1"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.3" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.15" length="0.6"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0.3" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.15" length="0.6"/>
      </geometry>
    </collision>
  </link>

  <!-- Joint connecting base to torso -->
  <joint name="base_to_torso" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>
</robot>
```

## 3. Defining Complex Joint Structures

### Humanoid Joint Types and Configurations

Humanoid robots require various joint types to achieve human-like movement:

#### Revolute Joints (Rotational)
```xml
<!-- Example: Hip joint -->
<joint name="left_hip_yaw" type="revolute">
  <parent link="torso"/>
  <child link="left_thigh"/>
  <origin xyz="0.05 -0.15 -0.3" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.57" upper="1.57" effort="100" velocity="3.14"/>
  <dynamics damping="0.1" friction="0.0"/>
</joint>
```

#### Continuous Joints (Unlimited Rotation)
```xml
<!-- Example: Head pan joint -->
<joint name="head_pan" type="continuous">
  <parent link="neck"/>
  <child link="head"/>
  <origin xyz="0 0 0.1" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <dynamics damping="0.1"/>
</joint>
```

#### Prismatic Joints (Linear Motion)
```xml
<!-- Example: Linear actuator -->
<joint name="linear_actuator" type="prismatic">
  <parent link="base_link"/>
  <child link="actuator_link"/>
  <origin xyz="0 0 0.1" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="0" upper="0.1" effort="100" velocity="1.0"/>
</joint>
```

## 4. Complete Humanoid URDF Example

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Materials -->
  <material name="blue">
    <color rgba="0.0 0.0 0.8 1.0"/>
  </material>
  <material name="red">
    <color rgba="0.8 0.0 0.0 1.0"/>
  </material>
  <material name="white">
    <color rgba="1.0 1.0 1.0 1.0"/>
  </material>

  <!-- Base link -->
  <link name="base_link">
    <inertial>
      <mass value="0.1"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Torso -->
  <link name="torso">
    <inertial>
      <mass value="5.0"/>
      <origin xyz="0 0 0.3" rpy="0 0 0"/>
      <inertia ixx="0.5" ixy="0" ixz="0" iyy="0.5" iyz="0" izz="0.1"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.3" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.15" length="0.6"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.3" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.15" length="0.6"/>
      </geometry>
    </collision>
  </link>

  <!-- Neck -->
  <link name="neck">
    <inertial>
      <mass value="0.5"/>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
    </collision>
  </link>

  <!-- Head -->
  <link name="head">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0.075" rpy="0 0 0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.075" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.15"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.075" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.15"/>
      </geometry>
    </collision>
  </link>

  <!-- Left Arm Links -->
  <link name="left_shoulder">
    <inertial>
      <mass value="0.5"/>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
    </collision>
  </link>

  <link name="left_upper_arm">
    <inertial>
      <mass value="0.8"/>
      <origin xyz="0 0 -0.1" rpy="0 0 0"/>
      <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.001"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.1" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.2"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.1" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.2"/>
      </geometry>
    </collision>
  </link>

  <!-- Joints -->
  <joint name="torso_to_neck" type="revolute">
    <parent link="torso"/>
    <child link="neck"/>
    <origin xyz="0 0 0.6" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.5" upper="0.5" effort="10" velocity="1.0"/>
  </joint>

  <joint name="neck_to_head" type="revolute">
    <parent link="neck"/>
    <child link="head"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="10" velocity="1.0"/>
  </joint>

  <joint name="torso_to_left_shoulder" type="revolute">
    <parent link="torso"/>
    <child link="left_shoulder"/>
    <origin xyz="0.15 0 0.4" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="1.0"/>
  </joint>

  <joint name="left_shoulder_to_upper_arm" type="revolute">
    <parent link="left_shoulder"/>
    <child link="left_upper_arm"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="1.0"/>
  </joint>
</robot>
```

## 5. Xacro for Complex Humanoid Models

For complex humanoid robots, Xacro (XML Macros) helps manage complexity:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_with_xacro">

  <!-- Properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="torso_mass" value="5.0" />
  <xacro:property name="arm_mass" value="0.8" />

  <!-- Macro for arm links -->
  <xacro:macro name="arm_chain" params="side parent_link position">
    <link name="${side}_shoulder">
      <inertial>
        <mass value="0.5"/>
        <origin xyz="0 0 0.05" rpy="0 0 0"/>
        <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
      </inertial>
      <visual>
        <origin xyz="0 0 0.05" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="0.05" length="0.1"/>
        </geometry>
      </visual>
    </link>

    <joint name="${parent_link}_to_${side}_shoulder" type="revolute">
      <parent link="${parent_link}"/>
      <child link="${side}_shoulder"/>
      <origin xyz="${position}" rpy="0 0 0"/>
      <axis xyz="1 0 0"/>
      <limit lower="${-M_PI/2}" upper="${M_PI/2}" effort="10" velocity="1.0"/>
    </joint>
  </xacro:macro>

  <!-- Use the macro -->
  <xacro:arm_chain side="left" parent_link="torso" position="0.15 0 0.4"/>
  <xacro:arm_chain side="right" parent_link="torso" position="-0.15 0 0.4"/>

</robot>
```

## 6. URDF Validation and Troubleshooting

### Common URDF Issues:
- **Missing joints**: Every link except the root must be connected by a joint
- **Invalid kinematic loops**: URDF cannot have closed loops (use transmissions for complex mechanisms)
- **Incorrect inertial properties**: Can cause simulation instability
- **Invalid joint limits**: Should not exceed physical capabilities

### Validation Tools:
```bash
# Check URDF validity
check_urdf /path/to/robot.urdf

# Visualize URDF
urdf_to_graphiz /path/to/robot.urdf
```

## 7. Integration with ROS 2 Systems

### Robot State Publisher
```xml
<!-- In launch file -->
<node pkg="robot_state_publisher" exec="robot_state_publisher" name="robot_state_publisher">
  <param name="robot_description" value="$(var robot_description)"/>
</node>
```

### Joint State Publisher for Visualization
```xml
<!-- For simulation -->
<node pkg="joint_state_publisher" exec="joint_state_publisher" name="joint_state_publisher">
  <param name="use_gui" value="true"/>
</node>
```

## 8. Best Practices for Humanoid URDF

- **Start simple**: Begin with basic shapes and add complexity gradually
- **Realistic inertial properties**: Use CAD models or approximate calculations
- **Proper joint limits**: Reflect actual hardware capabilities
- **Consistent naming**: Use descriptive, consistent names for links and joints
- **Modular design**: Use Xacro macros for repetitive structures
- **Validation**: Regularly validate URDF with tools

## Weekly Schedule Focus (Weeks 3-5)
During Weeks 3-5, we will focus on:
- Understanding URDF (Unified Robot Description Format) for humanoids
- Creating complex kinematic structures
- Validating and troubleshooting URDF models
- Integration with ROS 2 simulation systems

## Resources
- [URDF Documentation](http://wiki.ros.org/urdf)
- [URDF Tutorials](http://wiki.ros.org/urdf/Tutorials)
- [Xacro Documentation](http://wiki.ros.org/xacro)
- [Robot State Publisher](http://wiki.ros.org/robot_state_publisher)
