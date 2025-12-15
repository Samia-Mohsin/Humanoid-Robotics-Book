# Chapter 3: Physics Simulation

## Overview
This chapter covers the physics simulation aspects of Gazebo, focusing on realistic modeling of gravity, collisions, and dynamics for humanoid robots. Students will learn to configure physics parameters that accurately represent real-world physics, enabling reliable sim-to-real transfer of control algorithms.

## Learning Objectives
By the end of this chapter, students will be able to:
- Configure physics engines for realistic humanoid simulation
- Set appropriate collision detection and response parameters
- Model contact dynamics for stable humanoid locomotion
- Optimize physics parameters for computational efficiency
- Validate physics simulation against real-world behavior

## 1. Introduction to Physics Simulation in Gazebo

Physics simulation is the cornerstone of realistic robot simulation. For humanoid robots, accurate physics modeling is essential for:
- **Stability**: Proper balance and locomotion behaviors
- **Contact dynamics**: Realistic interaction with environment
- **Control validation**: Reliable testing of control algorithms
- **Safety**: Predicting robot behavior before real-world deployment

### Physics Engine Options:
- **ODE (Open Dynamics Engine)**: Default engine, good balance of speed and accuracy
- **Bullet**: Good for complex collision detection
- **Simbody**: High-fidelity multibody dynamics
- **DART**: Advanced constraint handling

## 2. Physics World Configuration

### World Physics Settings
```xml
<!-- worlds/humanoid_physics.world -->
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="humanoid_physics_world">
    <!-- Include standard models -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Physics engine configuration -->
    <physics name="humanoid_physics" type="ode">
      <!-- Time step for physics updates -->
      <max_step_size>0.001</max_step_size>

      <!-- Real-time update rate (steps per second) -->
      <real_time_update_rate>1000</real_time_update_rate>

      <!-- Real-time factor (1.0 = real-time, >1.0 = faster than real-time) -->
      <real_time_factor>1.0</real_time_factor>

      <!-- Physics engine parameters -->
      <ode>
        <!-- Solver parameters -->
        <solver>
          <type>quick</type>
          <iters>50</iters>
          <sor>1.3</sor>
          <use_dynamic_moi_rescaling>0</use_dynamic_moi_rescaling>
        </solver>

        <!-- Constraint parameters -->
        <constraints>
          <cfm>0.000001</cfm>
          <erp>0.2</erp>
          <contact_max_correcting_vel>100</contact_max_correcting_vel>
          <contact_surface_layer>0.001</contact_surface_layer>
        </constraints>
      </ode>
    </physics>
  </world>
</sdf>
```

### Physics Parameters Explained:
- **max_step_size**: Time step for physics integration (smaller = more accurate but slower)
- **real_time_update_rate**: How many physics steps per second of simulation
- **real_time_factor**: Desired speed relative to real time
- **iters**: Number of iterations for constraint solver (more = more stable but slower)
- **sor**: Successive Over-Relaxation parameter (affects convergence)
- **cfm**: Constraint Force Mixing (affects constraint stiffness)
- **erp**: Error Reduction Parameter (affects error correction)

## 3. Collision Detection and Response

### Collision Geometry Types
```xml
<!-- Different collision geometries for different purposes -->
<link name="collision_link">
  <!-- Box collision for simple rectangular shapes -->
  <collision name="box_collision">
    <geometry>
      <box>
        <size>0.1 0.1 0.1</size>
      </box>
    </geometry>
  </collision>

  <!-- Cylinder collision for limbs -->
  <collision name="cylinder_collision">
    <geometry>
      <cylinder>
        <radius>0.05</radius>
        <length>0.3</length>
      </cylinder>
    </geometry>
  </collision>

  <!-- Sphere collision for spherical joints -->
  <collision name="sphere_collision">
    <geometry>
      <sphere>
        <radius>0.05</radius>
      </sphere>
    </geometry>
  </collision>

  <!-- Mesh collision for complex shapes -->
  <collision name="mesh_collision">
    <geometry>
      <mesh>
        <uri>model://humanoid/meshes/complex_shape.stl</uri>
      </mesh>
    </geometry>
  </collision>
</link>
```

### Collision Properties for Humanoid Robots
```xml
<!-- Gazebo collision properties -->
<gazebo reference="left_foot">
  <collision name="left_foot_collision">
    <surface>
      <friction>
        <!-- ODE friction model -->
        <ode>
          <mu>0.8</mu>      <!-- Primary friction coefficient -->
          <mu2>0.8</mu2>    <!-- Secondary friction coefficient -->
          <fdir1>0 0 1</fdir1>  <!-- Friction direction -->
        </ode>
        <!-- Bullet friction model -->
        <bullet>
          <friction>0.8</friction>
          <friction2>0.8</friction2>
          <fdir1>0 0 1</fdir1>
          <rolling_friction>0.01</rolling_friction>
        </bullet>
      </friction>
      <bounce>
        <restitution_coefficient>0.1</restitution_coefficient>
        <threshold>100000</threshold>
      </bounce>
      <contact>
        <ode>
          <max_vel>100.0</max_vel>        <!-- Maximum contact velocity -->
          <min_depth>0.001</min_depth>     <!-- Minimum contact depth -->
        </ode>
      </contact>
    </surface>
  </collision>
</gazebo>
```

## 4. Inertial Properties for Humanoid Robots

### Accurate Inertial Modeling
```xml
<!-- Proper inertial properties for humanoid links -->
<link name="left_thigh">
  <inertial>
    <!-- Mass of the link in kg -->
    <mass>1.5</mass>

    <!-- Origin of the inertial reference frame -->
    <origin xyz="0 0 -0.15" rpy="0 0 0"/>

    <!-- Inertia matrix (symmetric 3x3 matrix) -->
    <inertia>
      <!-- Moments of inertia -->
      <ixx>0.01</ixx>
      <iyy>0.01</iyy>
      <izz>0.002</izz>
      <!-- Products of inertia -->
      <ixy>0.0</ixy>
      <ixz>0.0</ixz>
      <iyz>0.0</iyz>
    </inertia>
  </inertial>

  <visual>
    <origin xyz="0 0 -0.15" rpy="0 0 0"/>
    <geometry>
      <cylinder radius="0.06" length="0.3"/>
    </geometry>
  </visual>

  <collision>
    <origin xyz="0 0 -0.15" rpy="0 0 0"/>
    <geometry>
      <cylinder radius="0.06" length="0.3"/>
    </geometry>
  </collision>
</link>
```

### Calculating Inertial Properties
For complex humanoid body parts, inertial properties can be calculated using CAD software or approximated as combinations of simple geometric shapes:

```python
# Example: Calculate inertial properties for a humanoid limb
import numpy as np

def calculate_cylinder_inertia(mass, radius, length):
    """
    Calculate inertia tensor for a solid cylinder
    Ixx = Iyy = m/12 * (3*r² + h²)
    Izz = m/2 * r²
    """
    ixx = iyy = mass/12.0 * (3*radius**2 + length**2)
    izz = mass/2.0 * radius**2

    return np.array([[ixx, 0, 0],
                     [0, iyy, 0],
                     [0, 0, izz]])

def calculate_box_inertia(mass, width, depth, height):
    """
    Calculate inertia tensor for a solid box
    Ixx = m/12 * (h² + d²)
    Iyy = m/12 * (w² + h²)
    Izz = m/12 * (w² + d²)
    """
    ixx = mass/12.0 * (height**2 + depth**2)
    iyy = mass/12.0 * (width**2 + height**2)
    izz = mass/12.0 * (width**2 + depth**2)

    return np.array([[ixx, 0, 0],
                     [0, iyy, 0],
                     [0, 0, izz]])
```

## 5. Advanced Physics for Humanoid Locomotion

### Contact Stabilization for Walking
```xml
<!-- Physics parameters optimized for humanoid walking -->
<gazebo reference="left_foot">
  <surface>
    <contact>
      <ode>
        <!-- Increase max_vel for better contact handling during walking -->
        <max_vel>10.0</max_vel>
        <!-- Small min_depth for precise contact detection -->
        <min_depth>0.0001</min_depth>
      </ode>
    </contact>
    <friction>
      <ode>
        <!-- High friction for stable walking -->
        <mu>1.0</mu>
        <mu2>1.0</mu2>
      </ode>
    </friction>
  </surface>
</gazebo>

<!-- Global physics for humanoid walking -->
<physics name="humanoid_locomotion" type="ode">
  <max_step_size>0.001</max_step_size>
  <real_time_update_rate>1000</real_time_update_rate>
  <real_time_factor>1.0</real_time_factor>
  <ode>
    <solver>
      <type>quick</type>
      <!-- Increase iterations for stable contact handling -->
      <iters>200</iters>
      <sor>1.2</sor>
    </solver>
    <constraints>
      <!-- Stiffer constraints for precise control -->
      <cfm>1e-5</cfm>
      <erp>0.1</erp>
      <contact_max_correcting_vel>10.0</contact_max_correcting_vel>
      <contact_surface_layer>0.0005</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

### Balance and Stability Parameters
```xml
<!-- Physics configuration for balance control -->
<gazebo reference="torso">
  <surface>
    <contact>
      <ode>
        <!-- Higher max_vel for torso to handle balance corrections -->
        <max_vel>100.0</max_vel>
        <min_depth>0.001</min_depth>
      </ode>
    </contact>
  </surface>
</gazebo>

<!-- Center of mass considerations -->
<link name="torso">
  <inertial>
    <!-- Place center of mass appropriately for humanoid balance -->
    <mass>5.0</mass>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>  <!-- Slightly above hip level -->
    <inertia>
      <ixx>0.3</ixx>
      <iyy>0.3</iyy>
      <izz>0.1</izz>
    </inertia>
  </inertial>
</link>
```

## 6. Physics Performance Optimization

### Balancing Accuracy and Performance
```xml
<!-- Optimized physics for different simulation needs -->
<physics name="fast_physics" type="ode">
  <!-- For rapid prototyping -->
  <max_step_size>0.01</max_step_size>
  <real_time_update_rate>100</real_time_update_rate>
  <ode>
    <solver>
      <iters>20</iters>
    </solver>
    <constraints>
      <cfm>0.001</cfm>
      <erp>0.5</erp>
    </constraints>
  </ode>
</physics>

<physics name="accurate_physics" type="ode">
  <!-- For final validation -->
  <max_step_size>0.0005</max_step_size>
  <real_time_update_rate>2000</real_time_update_rate>
  <ode>
    <solver>
      <iters>200</iters>
    </solver>
    <constraints>
      <cfm>1e-6</cfm>
      <erp>0.05</erp>
    </constraints>
  </ode>
</physics>
```

### Adaptive Physics Configuration
```python
# Example: Python script to adjust physics parameters based on simulation needs
import xml.etree.ElementTree as ET

def adjust_physics_for_humanoid_locomotion(world_file, locomotion_type="walking"):
    """
    Adjust physics parameters for different types of humanoid locomotion
    """
    tree = ET.parse(world_file)
    root = tree.getroot()

    # Find physics element
    physics_elem = root.find('physics')

    if locomotion_type == "walking":
        # More stable parameters for walking
        max_step_size = physics_elem.find('max_step_size')
        max_step_size.text = "0.001"

        solver = physics_elem.find('.//solver')
        iters = solver.find('iters')
        iters.text = "150"  # More iterations for stability

        constraints = physics_elem.find('.//constraints')
        cfm = constraints.find('cfm')
        cfm.text = "1e-5"  # Stiffer constraints
        erp = constraints.find('erp')
        erp.text = "0.1"   # Faster error correction

    elif locomotion_type == "running":
        # Parameters optimized for faster, less stable motion
        max_step_size = physics_elem.find('max_step_size')
        max_step_size.text = "0.0005"  # Smaller steps for accuracy

        solver = physics_elem.find('.//solver')
        iters = solver.find('iters')
        iters.text = "300"  # More iterations for complex contact

    tree.write(world_file)
    print(f"Physics adjusted for {locomotion_type}")
```

## 7. Validation and Tuning

### Physics Validation Techniques
```bash
# Compare simulation with real-world data
# 1. Record joint trajectories in simulation
ros2 topic echo /joint_states --field position > sim_joint_data.txt

# 2. Record same trajectories on real robot
# (real robot data collection)

# 3. Compare center of mass trajectories
ros2 run humanoid_control com_analyzer --sim-data sim_joint_data.txt --real-data real_joint_data.txt
```

### Common Physics Issues and Solutions:

1. **Robot falls through ground**:
   - Check that collision geometries exist for all links
   - Verify that inertial properties are properly defined
   - Ensure physics engine is running

2. **Unstable walking**:
   - Increase solver iterations
   - Adjust ERP and CFM values
   - Verify friction coefficients are realistic

3. **Joint limit violations**:
   - Check that joint limits are properly set in URDF
   - Verify that control commands respect limits
   - Adjust physics parameters for better constraint enforcement

4. **Performance issues**:
   - Reduce solver iterations (trade accuracy for speed)
   - Increase time step (trade accuracy for speed)
   - Simplify collision geometries

## 8. Advanced Physics Features

### Joint Friction and Damping
```xml
<!-- Adding friction and damping to joints for realistic behavior -->
<joint name="left_knee" type="revolute">
  <parent link="left_thigh"/>
  <child link="left_shin"/>
  <origin xyz="0 0 -0.3" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="-0.1" upper="2.5" effort="100" velocity="3.14"/>
  <!-- Joint dynamics for realistic movement -->
  <dynamics damping="0.5" friction="0.1"/>
</joint>

<!-- Gazebo-specific joint properties -->
<gazebo reference="left_knee">
  <provideFeedback>true</provideFeedback>
  <implicitSpringDamper>1</implicitSpringDamper>
  <springReference>0.0</springReference>
  <springStiffness>0.0</springStiffness>
</gazebo>
```

### Custom Physics Plugins
```cpp
// Example: Custom physics plugin for humanoid-specific physics
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>

namespace gazebo
{
  class HumanoidPhysicsPlugin : public WorldPlugin
  {
    public: void Load(physics::WorldPtr _world, sdf::ElementPtr _sdf)
    {
      // Custom physics for humanoid robots
      this->world = _world;

      // Connect to physics update event
      this->updateConnection = event::Events::ConnectWorldUpdateBegin(
          std::bind(&HumanoidPhysicsPlugin::OnUpdate, this));
    }

    public: void OnUpdate()
    {
      // Custom physics calculations for humanoid stability
      // This is a simplified example
    }

    private: physics::WorldPtr world;
    private: event::ConnectionPtr updateConnection;
  };

  GZ_REGISTER_WORLD_PLUGIN(HumanoidPhysicsPlugin)
}
```

## 9. Physics Debugging and Visualization

### Debugging Physics Issues
```bash
# Enable physics debugging in Gazebo
gzserver --verbose worlds/humanoid_physics.world

# Monitor physics statistics
gz stats

# Visualize contact forces
# In Gazebo GUI: View -> Contacts
```

### Physics Performance Monitoring
```python
# Physics performance monitoring node
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

class PhysicsMonitor(Node):
    def __init__(self):
        super().__init__('physics_monitor')

        # Publishers for physics metrics
        self.real_time_factor_pub = self.create_publisher(Float32, 'physics/real_time_factor', 10)
        self.sim_time_pub = self.create_publisher(Float32, 'physics/sim_time', 10)

        # Timer for monitoring
        self.monitor_timer = self.create_timer(1.0, self.monitor_physics)

        self.get_logger().info('Physics Monitor initialized')

    def monitor_physics(self):
        # In a real implementation, this would interface with Gazebo
        # to get physics performance metrics
        rtf_msg = Float32()
        rtf_msg.data = 1.0  # Placeholder value
        self.real_time_factor_pub.publish(rtf_msg)

def main(args=None):
    rclpy.init(args=args)
    monitor = PhysicsMonitor()

    try:
        rclpy.spin(monitor)
    except KeyboardInterrupt:
        monitor.get_logger().info('Shutting down physics monitor')
    finally:
        monitor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 10. Best Practices for Physics Simulation

### Performance Optimization:
- Use simplified collision meshes separate from visual meshes
- Balance time step size with accuracy requirements
- Adjust solver parameters based on simulation complexity
- Use appropriate friction and contact parameters

### Accuracy Considerations:
- Validate simulation results against real-world data
- Use realistic inertial properties
- Match physics parameters to real-world robot characteristics
- Test with various environmental conditions

### Stability Tips:
- Start with conservative physics parameters
- Gradually increase complexity and performance
- Monitor simulation stability during development
- Use appropriate damping and friction values

## Weekly Schedule Focus (Weeks 6-7)
During Weeks 6-7, we will focus on:
- Physics simulation and sensor simulation
- Simulating physics, gravity, and collisions in Gazebo
- Introduction to Unity for robot visualization
- Human-robot interaction in simulation environments

## Resources
- [Gazebo Physics Documentation](http://gazebosim.org/tutorials?tut=physics)
- [ODE User Manual](http://ode.org/wiki/index.php?title=Manual)
- [Physics Parameter Tuning Guide](http://gazebosim.org/tutorials?tut=physics_tuning)
- [Humanoid Robot Simulation Best Practices](https://humanoid-walk.readthedocs.io/)
