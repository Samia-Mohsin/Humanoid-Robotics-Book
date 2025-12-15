# Chapter 1: Gazebo Simulation Environment Setup

## Overview
This chapter covers the installation, configuration, and initial setup of the Gazebo simulation environment. Students will learn to create a complete simulation environment for humanoid robots, including physics simulation, sensor modeling, and visualization tools that form the "Digital Twin" of the physical robot.

## Learning Objectives
By the end of this chapter, students will be able to:
- Install and configure Gazebo simulation environment
- Set up the complete simulation workspace with ROS 2 integration
- Configure physics parameters for realistic humanoid simulation
- Create and test basic simulation environments
- Understand the relationship between simulation and real hardware

## 1. Introduction to Gazebo for Humanoid Robotics

Gazebo is a 3D simulation environment that enables testing of robotics applications without the need for physical hardware. For humanoid robots, Gazebo provides:
- **Physics simulation**: Realistic modeling of gravity, collisions, and dynamics
- **Sensor simulation**: Virtual sensors that mimic real hardware (LiDAR, cameras, IMUs)
- **Environment modeling**: Creation of complex worlds for testing
- **Hardware integration**: Seamless integration with ROS 2 and real hardware

### Why Gazebo for Humanoid Robotics?
- **Safety**: Test complex behaviors without risk of hardware damage
- **Cost-effectiveness**: Reduce need for physical prototypes
- **Repeatability**: Conduct controlled experiments with consistent conditions
- **Scalability**: Test multiple robots simultaneously
- **Speed**: Accelerate development through faster iteration cycles

## 2. System Requirements and Prerequisites

### Hardware Requirements for Gazebo Simulation:
- **GPU**: NVIDIA RTX 4070 Ti (12GB VRAM) or higher (required for Isaac Sim, recommended for Gazebo)
- **CPU**: Intel Core i7 (13th Gen+) or AMD Ryzen 9
- **RAM**: 64 GB DDR5 (32 GB minimum for basic simulations)
- **OS**: Ubuntu 22.04 LTS (recommended for ROS 2 compatibility)

### Software Prerequisites:
- Ubuntu 22.04 LTS (or ROS 2 supported OS)
- ROS 2 Iron/Iguana installation
- OpenGL 3.3+ capable graphics card
- Sufficient disk space (20+ GB recommended)

## 3. Installing Gazebo

### Installing Gazebo Garden (Recommended for ROS 2 Iron)
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Gazebo dependencies
sudo apt install wget lsb-release gnupg

# Add Gazebo repository
echo "deb [arch=amd64] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

# Install Gazebo signing key
wget https://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -

# Update package list
sudo apt update

# Install Gazebo Garden
sudo apt install gazebo-garden
```

### Installing ROS 2 Gazebo Integration
```bash
# Install ROS 2 Gazebo packages
sudo apt install ros-iron-gazebo-ros-pkgs ros-iron-gazebo-plugins ros-iron-gazebo-dev

# Install additional simulation tools
sudo apt install ros-iron-ros-gz ros-iron-ros-gz-bridge ros-iron-ros-gz-interfaces
```

## 4. Initial Gazebo Configuration

### Basic Gazebo Launch Test
```bash
# Launch Gazebo in standalone mode
gazebo

# Or launch with an empty world
gazebo --verbose worlds/empty.world
```

### Environment Variables Setup
```bash
# Add to ~/.bashrc for persistent configuration
echo 'export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:~/.gazebo/models:/usr/share/gazebo-11/models' >> ~/.bashrc
echo 'export GAZEBO_RESOURCE_PATH=$GAZEBO_RESOURCE_PATH:~/.gazebo/models:/usr/share/gazebo-11' >> ~/.bashrc
echo 'export GAZEBO_PLUGIN_PATH=$GAZEBO_PLUGIN_PATH:/usr/lib/x86_64-linux-gnu/gazebo-11/plugins' >> ~/.bashrc

# Source the changes
source ~/.bashrc
```

## 5. Gazebo with ROS 2 Integration

### Creating a Gazebo Workspace
```bash
# Create workspace
mkdir -p ~/gazebo_ws/src
cd ~/gazebo_ws

# Source ROS 2
source /opt/ros/iron/setup.bash

# Create a package for Gazebo models and worlds
cd src
ros2 pkg create --build-type ament_cmake humanoid_gazebo --dependencies gazebo_ros_pkgs
cd ..
colcon build --packages-select humanoid_gazebo
source install/setup.bash
```

### Basic Gazebo Launch with ROS 2
```xml
<!-- launch/gazebo.launch.py -->
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node

def generate_launch_description():
    # Gazebo launch file
    gazebo_launch = PathJoinSubstitution([
        FindPackageShare('gazebo_ros'),
        'launch',
        'gazebo.launch.py'
    ])

    # Include Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gazebo_launch),
        launch_arguments={
            'world': PathJoinSubstitution([
                FindPackageShare('humanoid_gazebo'),
                'worlds',
                'humanoid_lab.world'
            ])
        }.items()
    )

    return LaunchDescription([
        gazebo,
    ])
```

## 6. Creating Your First Humanoid Simulation Environment

### Basic World File (worlds/humanoid_lab.world)
```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="humanoid_lab">
    <!-- Include a ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Include lighting -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Define a simple room -->
    <model name="room_walls">
      <static>true</static>
      <link name="room_link">
        <visual name="room_visual">
          <geometry>
            <box>
              <size>10 10 3</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
          </material>
        </visual>
        <collision name="room_collision">
          <geometry>
            <box>
              <size>10 10 3</size>
            </box>
          </geometry>
        </collision>
      </link>
    </model>

    <!-- Physics parameters -->
    <physics name="1ms" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>
  </world>
</sdf>
```

## 7. Launching Gazebo with Your World

### Testing the Setup
```bash
# Source your workspace
cd ~/gazebo_ws
source install/setup.bash

# Launch your custom world
ros2 launch humanoid_gazebo gazebo.launch.py

# Or launch directly with Gazebo
gazebo --verbose worlds/humanoid_lab.world
```

## 8. Troubleshooting Common Setup Issues

### Graphics Issues
```bash
# If you encounter graphics errors, try:
export MESA_GL_VERSION_OVERRIDE=3.3
gazebo

# For NVIDIA GPU specific issues:
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
gazebo
```

### Common Error Solutions:
1. **"libgazebo.so not found"**: Check that Gazebo is properly installed and library paths are set
2. **Rendering issues**: Ensure graphics drivers are up to date and OpenGL is supported
3. **Performance issues**: Increase physics update rate or reduce visual complexity
4. **ROS 2 integration errors**: Verify that ROS 2 Gazebo packages are installed

### Verification Steps
```bash
# Check Gazebo version
gazebo --version

# Verify ROS 2 integration
ros2 run gazebo_ros spawn_entity.py -h

# Test basic launch
gazebo --verbose worlds/empty.world
```

## 9. Best Practices for Gazebo Setup

### Performance Optimization:
- Use simpler collision meshes for better performance
- Adjust physics update rates based on simulation needs
- Limit the number of active sensors during initial testing
- Use appropriate world complexity for your hardware

### Organization:
- Create separate packages for different robot models
- Organize worlds, models, and launch files systematically
- Use version control for simulation assets
- Document physics parameters and configurations

## Weekly Schedule Focus (Weeks 6-7)
During Weeks 6-7, we will focus on:
- Gazebo simulation environment setup
- Basic world creation and testing
- Integration with ROS 2 systems
- Physics simulation and sensor modeling

## Resources
- [Gazebo Documentation](https://gazebosim.org/)
- [ROS 2 Gazebo Integration](http://gazebosim.org/tutorials?cat=connect_ros)
- [Gazebo Tutorials](https://classic.gazebosim.org/tutorials)
- [Ubuntu Installation Guide](https://gazebosim.org/docs/latest/install_ubuntu/)
