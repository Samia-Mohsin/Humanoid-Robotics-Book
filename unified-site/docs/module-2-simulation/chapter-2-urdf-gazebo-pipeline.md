# Chapter 2: URDF-Gazebo Pipeline

## Overview
This chapter covers the integration of URDF (Unified Robot Description Format) with Gazebo simulation, creating a seamless pipeline from robot description to simulation. Students will learn to enhance their URDF models with Gazebo-specific tags to enable physics simulation, sensor modeling, and visualization in the digital twin environment.

## Learning Objectives
By the end of this chapter, students will be able to:
- Enhance URDF models with Gazebo-specific extensions
- Configure physics properties for realistic simulation
- Add sensor models to URDF for Gazebo simulation
- Set up the complete URDF-Gazebo workflow
- Troubleshoot common integration issues

## 1. Introduction to URDF-Gazebo Integration

The URDF-Gazebo pipeline is crucial for creating accurate digital twins of humanoid robots. While URDF describes the robot's kinematic structure, Gazebo tags extend this description with:

- **Physics properties**: Mass, inertia, friction, and collision characteristics
- **Visual properties**: Materials, textures, and rendering options
- **Sensor models**: Virtual sensors that mimic real hardware
- **Actuator models**: Motor characteristics and control interfaces

### Key Benefits of the Pipeline:
- **Single source of truth**: One robot description for both simulation and real hardware
- **Consistent interfaces**: Same ROS topics and services in simulation and reality
- **Efficient development**: Test and debug in simulation before hardware deployment

## 2. Gazebo-Specific URDF Extensions

### Basic Gazebo Tags Structure
```xml
<!-- Enhanced URDF with Gazebo-specific tags -->
<?xml version="1.0"?>
<robot name="humanoid_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Links defined as in standard URDF -->
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

  <!-- Gazebo-specific extensions -->
  <gazebo reference="base_link">
    <material>Gazebo/Grey</material>
    <mu1>0.2</mu1>
    <mu2>0.2</mu2>
    <kp>1000000.0</kp>
    <kd>100.0</kd>
  </gazebo>
</robot>
```

## 3. Physics Configuration in URDF

### Adding Physics Properties
```xml
<!-- Complete example with physics properties -->
<?xml version="1.0"?>
<robot name="humanoid_with_physics">
  <!-- Link with detailed physics properties -->
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
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.3" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.15" length="0.6"/>
      </geometry>
    </collision>
  </link>

  <!-- Gazebo physics extensions -->
  <gazebo reference="torso">
    <material>Gazebo/White</material>
    <!-- Friction coefficients -->
    <mu1>0.8</mu1>
    <mu2>0.8</mu2>
    <!-- Contact parameters -->
    <kp>10000000.0</kp>  <!-- Spring stiffness -->
    <kd>1000.0</kd>     <!-- Damping coefficient -->
    <!-- ODE friction parameters -->
    <fdir1>0 0 1</fdir1>
    <maxVel>100.0</maxVel>
    <minDepth>0.001</minDepth>
  </gazebo>
</robot>
```

### Joint Physics Configuration
```xml
<!-- Joint with physics properties -->
<joint name="left_hip" type="revolute">
  <parent link="torso"/>
  <child link="left_thigh"/>
  <origin xyz="0.1 -0.15 -0.3" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.57" upper="1.57" effort="100" velocity="3.14"/>
  <dynamics damping="0.1" friction="0.01"/>
</joint>

<!-- Gazebo-specific joint configuration -->
<gazebo reference="left_hip">
  <provideFeedback>true</provideFeedback>
  <implicitSpringDamper>1</implicitSpringDamper>
</gazebo>
```

## 4. Sensor Integration in URDF

### Adding Camera Sensors
```xml
<!-- Camera sensor in URDF -->
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

<!-- Gazebo camera sensor configuration -->
<gazebo reference="camera_link">
  <sensor name="camera1" type="camera">
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
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>camera_link</frame_name>
      <topic_name>head_camera/image_raw</topic_name>
      <hack_baseline>0.07</hack_baseline>
    </plugin>
  </sensor>
</gazebo>
```

### Adding LiDAR Sensors
```xml
<!-- LiDAR sensor configuration -->
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
  <origin xyz="0 0 0.2" rpy="0 0 0"/>
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
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
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
        <namespace>laser</namespace>
        <remapping>~/out:=scan</remapping>
      </ros>
      <output_type>sensor_msgs/LaserScan</output_type>
    </plugin>
  </sensor>
</gazebo>
```

### Adding IMU Sensors
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
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </z>
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
    <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
      <ros>
        <namespace>imu</namespace>
        <remapping>~/out:=data</remapping>
      </ros>
      <initial_orientation_as_reference>false</initial_orientation_as_reference>
    </plugin>
  </sensor>
</gazebo>
```

## 5. Complete Humanoid Robot Example

### Full URDF with Gazebo Integration
```xml
<?xml version="1.0"?>
<robot name="simple_humanoid_gazebo" xmlns:xacro="http://www.ros.org/wiki/xacro">
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

  <!-- Joints -->
  <joint name="torso_to_head" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.6" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="10" velocity="1.0"/>
  </joint>

  <!-- Gazebo-specific configurations -->
  <gazebo reference="torso">
    <material>Gazebo/White</material>
    <mu1>0.8</mu1>
    <mu2>0.8</mu2>
    <kp>10000000.0</kp>
    <kd>1000.0</kd>
  </gazebo>

  <gazebo reference="head">
    <material>Gazebo/White</material>
    <mu1>0.8</mu1>
    <mu2>0.8</mu2>
    <kp>10000000.0</kp>
    <kd>1000.0</kd>
  </gazebo>

  <!-- IMU sensor in head -->
  <gazebo reference="head">
    <sensor name="head_imu" type="imu">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <imu>
        <angular_velocity>
          <x>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>2e-4</stddev>
            </noise>
          </x>
          <y>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>2e-4</stddev>
            </noise>
          </y>
          <z>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>2e-4</stddev>
            </noise>
          </z>
        </angular_velocity>
        <linear_acceleration>
          <x>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>1.7e-2</stddev>
            </noise>
          </x>
          <y>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>1.7e-2</stddev>
            </noise>
          </y>
          <z>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>1.7e-2</stddev>
            </noise>
          </z>
        </linear_acceleration>
      </imu>
      <plugin name="head_imu_plugin" filename="libgazebo_ros_imu.so">
        <ros>
          <namespace>imu</namespace>
          <remapping>~/out:=data</remapping>
        </ros>
        <initial_orientation_as_reference>false</initial_orientation_as_reference>
      </plugin>
    </sensor>
  </gazebo>

  <!-- Camera in head -->
  <gazebo reference="head">
    <sensor name="head_camera" type="camera">
      <always_on>true</always_on>
      <update_rate>30.0</update_rate>
      <camera name="head_camera">
        <horizontal_fov>1.3962634</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.1</near>
          <far>100</far>
        </clip>
      </camera>
      <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
        <frame_name>head</frame_name>
        <topic_name>head_camera/image_raw</topic_name>
      </plugin>
    </sensor>
  </gazebo>
</robot>
```

## 6. Gazebo Plugins for Control

### Joint State Publisher Plugin
```xml
<!-- Add to your URDF -->
<gazebo>
  <plugin name="joint_state_publisher" filename="libgazebo_ros_joint_state_publisher.so">
    <ros>
      <namespace>/robot</namespace>
    </ros>
    <update_rate>30</update_rate>
    <joint_name>left_hip</joint_name>
    <joint_name>right_hip</joint_name>
    <joint_name>left_knee</joint_name>
    <joint_name>right_knee</joint_name>
  </plugin>
</gazebo>
```

### Joint Position Controller Plugin
```xml
<!-- Joint controller plugin -->
<gazebo>
  <plugin name="position_controller" filename="libgazebo_ros_joint_position.so">
    <ros>
      <namespace>/robot</namespace>
      <remapping>cmd:=/left_hip_position/command</remapping>
      <remapping>state:=/left_hip_position/state</remapping>
    </ros>
    <joint_name>left_hip</joint_name>
    <update_rate>100</update_rate>
    <command_topic>command</command_topic>
    <state_topic>state</state_topic>
    <feedback_topic>feedback</feedback_topic>
  </plugin>
</gazebo>
```

## 7. Launch Files for URDF-Gazebo Integration

### Robot State Publisher and Gazebo Launch
```python
# launch/robot_spawn.launch.py
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
import os

def generate_launch_description():
    # Launch configuration
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    robot_description_path = LaunchConfiguration('robot_description_path')

    # Get the path to the URDF file
    robot_description_path_default = PathJoinSubstitution([
        FindPackageShare('humanoid_gazebo'),
        'models',
        'humanoid',
        'urdf',
        'humanoid_gazebo.urdf'
    ])

    # Robot state publisher node
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': PathJoinSubstitution([
                FindPackageShare('humanoid_gazebo'),
                'models',
                'humanoid',
                'urdf',
                'humanoid_gazebo.urdf'
            ])
        }]
    )

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
            'use_sim_time': use_sim_time
        }.items()
    )

    # Spawn entity node
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

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation clock if true'
        ),
        robot_state_publisher,
        gazebo,
        spawn_entity,
    ])
```

## 8. Testing the URDF-Gazebo Pipeline

### Verification Commands
```bash
# Check URDF validity with Gazebo tags
check_urdf /path/to/robot.urdf

# Launch the robot in Gazebo
ros2 launch humanoid_gazebo robot_spawn.launch.py

# Check that sensors are publishing data
ros2 topic list | grep -E "(camera|imu|scan)"

# Monitor joint states
ros2 topic echo /joint_states --field name
```

## 9. Troubleshooting Common Issues

### Common Problems and Solutions:

1. **Robot falls through the ground**:
   - Check that inertial properties are properly defined
   - Verify that collision geometries are present
   - Ensure physics parameters (mu1, mu2, kp, kd) are set appropriately

2. **Sensors not publishing data**:
   - Verify plugin names and filenames are correct
   - Check that ROS topics are properly configured
   - Ensure update rates are reasonable

3. **Joints not responding to commands**:
   - Check that controller plugins are properly configured
   - Verify joint names match between URDF and controller
   - Ensure proper ROS topic remappings

4. **Performance issues**:
   - Reduce visual complexity of models
   - Adjust physics update rates
   - Limit the number of active sensors during testing

## 10. Best Practices for URDF-Gazebo Integration

### Performance Optimization:
- Use simplified collision meshes separate from visual meshes
- Set appropriate update rates for different sensors
- Use fixed joints where possible to reduce computational load
- Implement level-of-detail (LOD) for complex models

### Organization:
- Keep URDF files modular using xacro macros
- Separate visual, collision, and inertial properties clearly
- Use consistent naming conventions for joints and links
- Document custom Gazebo plugins and configurations

## Weekly Schedule Focus (Weeks 6-7)
During Weeks 6-7, we will focus on:
- URDF and SDF robot description formats
- Physics simulation and sensor simulation
- Integration of sensors in Gazebo simulation
- Human-robot interaction in simulation environments

## Resources
- [Gazebo URDF Integration](http://gazebosim.org/tutorials/?tut=ros2_overview)
- [URDF Specification](http://wiki.ros.org/urdf/XML)
- [Gazebo SDF Documentation](http://sdformat.org/spec)
- [ROS 2 Control Integration](https://control.ros.org/)
