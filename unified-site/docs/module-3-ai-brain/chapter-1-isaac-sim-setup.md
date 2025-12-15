# Chapter 1: NVIDIA Isaac Sim Setup

## Overview
This chapter covers the installation, configuration, and setup of NVIDIA Isaac Sim, a powerful simulation environment for robotics applications. Students will learn to create photorealistic simulation environments, generate synthetic data, and prepare for advanced perception and training tasks using Isaac Sim's capabilities for humanoid robots.

## Learning Objectives
By the end of this chapter, students will be able to:
- Install and configure NVIDIA Isaac Sim with proper hardware requirements
- Set up the Omniverse environment for robot simulation
- Create and configure robot assets for Isaac Sim
- Generate synthetic data for training perception systems
- Configure lighting and materials for photorealistic rendering
- Understand the integration between Isaac Sim and robotics frameworks

## 1. Introduction to NVIDIA Isaac Sim

NVIDIA Isaac Sim is a comprehensive simulation environment built on NVIDIA's Omniverse platform, designed specifically for robotics applications. For humanoid robots, Isaac Sim provides:

- **Photorealistic rendering**: Advanced lighting and materials for realistic perception training
- **Synthetic data generation**: Massive datasets for training computer vision models
- **Physics simulation**: Accurate physics with GPU acceleration
- **Sensor simulation**: Realistic camera, LiDAR, and other sensor models
- **AI training environments**: Ready-made environments for reinforcement learning

### Key Features for Humanoid Robotics:
- **Advanced rendering pipeline**: RTX-accelerated path tracing for realistic lighting
- **PhysX physics engine**: Accurate collision detection and response
- **Synthetic data generation**: Tools for creating labeled datasets
- **ROS/ROS2 integration**: Seamless communication with robotics frameworks
- **Cloud deployment**: Scalable simulation environments

## 2. Hardware Requirements and Prerequisites

### System Requirements for Isaac Sim:
- **GPU**: NVIDIA RTX 4070 Ti (12GB VRAM) or higher (required for Isaac Sim)
- **CPU**: Intel Core i7 (13th Gen+) or AMD Ryzen 9 with 16+ cores
- **RAM**: 64 GB DDR5 (32 GB minimum, but 64 GB recommended for complex scenes)
- **Storage**: 50+ GB SSD for Isaac Sim installation and assets
- **OS**: Ubuntu 22.04 LTS or Windows 10/11 (Linux recommended for robotics)

### GPU Requirements for Photorealistic Rendering:
- **Minimum**: NVIDIA RTX 3070 with 8GB VRAM
- **Recommended**: NVIDIA RTX 4080/4090 with 16GB+ VRAM
- **VRAM considerations**: Complex humanoid scenes may require 12GB+ VRAM

### Software Prerequisites:
- NVIDIA GPU drivers (latest Game Ready or Studio drivers)
- CUDA 11.8 or later
- NVIDIA Omniverse Launcher
- Python 3.8-3.11
- ROS 2 Humble/Iron (for robotics integration)

## 3. Installing NVIDIA Isaac Sim

### Installing via Omniverse Launcher:
```bash
# 1. Download and install Omniverse Launcher from NVIDIA Developer website
# 2. Launch Omniverse Launcher and sign in with your NVIDIA Developer account
# 3. Install Isaac Sim from the Apps section

# The following steps assume Isaac Sim is installed via Omniverse Launcher
# Navigate to the Isaac Sim installation directory
cd ~/.local/share/ov/pkg/isaac_sim-[VERSION]/

# Verify installation
python -c "import omni; print('Isaac Sim installed successfully')"
```

### Alternative Installation via Docker:
```bash
# Pull Isaac Sim Docker image
docker pull nvcr.io/nvidia/isaac-sim:4.0.0

# Run Isaac Sim container
docker run --gpus all -it --rm \
  --network=host \
  --env NVIDIA_DISABLE_REQUIRE=1 \
  --volume $(pwd):/workspace/current \
  --volume ~/.Xauthority:/root/.Xauthority \
  --volume /tmp/.X11-unix:/tmp/.X11-unix \
  --device /dev/dri:/dev/dri \
  --device /dev/snd:/dev/snd \
  nvcr.io/nvidia/isaac-sim:4.0.0
```

## 4. Initial Isaac Sim Configuration

### Launching Isaac Sim:
```bash
# Launch Isaac Sim via Omniverse Launcher (GUI method)
# Or launch from command line:
./python.sh ./apps/omni.isaac.sim.python.sh
```

### Basic Isaac Sim Python Script:
```python
# hello_isaac_sim.py
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
import carb

# Create a world instance
my_world = World(stage_units_in_meters=1.0)

# Add a simple robot to the stage
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
else:
    # Add a simple robot asset
    add_reference_to_stage(
        usd_path=assets_root_path + "/Isaac/Robots/Franka/franka.usd",
        prim_path="/World/Franka"
    )

# Play the simulation
my_world.play()

# Step the world for a few iterations
for i in range(100):
    my_world.step(render=True)

# Stop the simulation
my_world.stop()
```

## 5. Setting Up Humanoid Robot Assets

### Creating a Humanoid Robot USD Asset:
```python
# humanoid_setup.py
import omni
from pxr import Usd, UsdGeom, Gf, Sdf, UsdPhysics, PhysxSchema
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.rotations import euler_angles_to_quat
import numpy as np

def create_humanoid_robot_stage(robot_name="/World/HumanoidRobot"):
    """
    Create a basic humanoid robot in Isaac Sim
    This is a simplified example - real humanoid robots would have more complex joint structures
    """
    # Get the current stage
    stage = omni.usd.get_context().get_stage()

    # Create a prim for the robot
    robot_prim = UsdGeom.Xform.Define(stage, robot_name)

    # Create the root/base link
    root_path = f"{robot_name}/BaseLink"
    root_prim = UsdGeom.Xform.Define(stage, root_path)

    # Add basic geometry to root
    UsdGeom.Cube.Define(stage, f"{root_path}/Geometry")
    cube_geom = UsdGeom.Cube.Get(stage, f"{root_path}/Geometry")
    cube_geom.GetSizeAttr().Set(0.2)  # 20cm cube

    # Create torso
    torso_path = f"{robot_name}/Torso"
    torso_prim = UsdGeom.Xform.Define(stage, torso_path)
    torso_prim.AddTranslateOp().Set(Gf.Vec3f(0, 0, 0.3))  # Position torso above base

    # Add torso geometry
    UsdGeom.Cylinder.Define(stage, f"{torso_path}/Geometry")
    cylinder_geom = UsdGeom.Cylinder.Get(stage, f"{torso_path}/Geometry")
    cylinder_geom.GetRadiusAttr().Set(0.15)  # 15cm radius
    cylinder_geom.GetHeightAttr().Set(0.6)   # 60cm height

    # Create head
    head_path = f"{robot_name}/Head"
    head_prim = UsdGeom.Xform.Define(stage, head_path)
    head_prim.AddTranslateOp().Set(Gf.Vec3f(0, 0, 0.6))  # Position head above torso

    # Add head geometry
    UsdGeom.Sphere.Define(stage, f"{head_path}/Geometry")
    sphere_geom = UsdGeom.Sphere.Get(stage, f"{head_path}/Geometry")
    sphere_geom.GetRadiusAttr().Set(0.15)  # 15cm radius

    # Create left arm
    left_arm_path = f"{robot_name}/LeftArm"
    left_arm_prim = UsdGeom.Xform.Define(stage, left_arm_path)
    left_arm_prim.AddTranslateOp().Set(Gf.Vec3f(0.2, 0, 0.3))  # Position left of torso

    # Add left arm geometry
    UsdGeom.Capsule.Define(stage, f"{left_arm_path}/Geometry")
    capsule_geom = UsdGeom.Capsule.Get(stage, f"{left_arm_path}/Geometry")
    capsule_geom.GetRadiusAttr().Set(0.05)  # 5cm radius
    capsule_geom.GetHeightAttr().Set(0.4)   # 40cm height

    # Create right arm
    right_arm_path = f"{robot_name}/RightArm"
    right_arm_prim = UsdGeom.Xform.Define(stage, right_arm_path)
    right_arm_prim.AddTranslateOp().Set(Gf.Vec3f(-0.2, 0, 0.3))  # Position right of torso

    # Add right arm geometry
    UsdGeom.Capsule.Define(stage, f"{right_arm_path}/Geometry")
    UsdGeom.Capsule.Get(stage, f"{right_arm_path}/Geometry")

    # Create left leg
    left_leg_path = f"{robot_name}/LeftLeg"
    left_leg_prim = UsdGeom.Xform.Define(stage, left_leg_path)
    left_leg_prim.AddTranslateOp().Set(Gf.Vec3f(0.1, 0, -0.3))  # Position below torso

    # Add left leg geometry
    UsdGeom.Capsule.Define(stage, f"{left_leg_path}/Geometry")
    UsdGeom.Capsule.Get(stage, f"{left_leg_path}/Geometry")

    # Create right leg
    right_leg_path = f"{robot_name}/RightLeg"
    right_leg_prim = UsdGeom.Xform.Define(stage, right_leg_path)
    right_leg_prim.AddTranslateOp().Set(Gf.Vec3f(-0.1, 0, -0.3))  # Position below torso

    # Add right leg geometry
    UsdGeom.Capsule.Define(stage, f"{right_leg_path}/Geometry")
    UsdGeom.Capsule.Get(stage, f"{right_leg_path}/Geometry")

    print(f"Created basic humanoid robot at {robot_name}")

def setup_physics_for_humanoid(robot_path="/World/HumanoidRobot"):
    """
    Add physics properties to the humanoid robot
    """
    stage = omni.usd.get_context().get_stage()

    # Add rigid body properties to each link
    links = ["BaseLink", "Torso", "Head", "LeftArm", "RightArm", "LeftLeg", "RightLeg"]

    for link in links:
        link_path = f"{robot_path}/{link}"
        link_prim = stage.GetPrimAtPath(link_path)

        # Add rigid body API
        UsdPhysics.RigidBodyAPI.Apply(link_prim)

        # Add mass and inertia
        rigid_body_api = UsdPhysics.RigidBodyAPI.Get(stage, link_path)
        rigid_body_api.CreateMassAttr(1.0)  # 1kg for simplicity

        # Add collision API
        UsdPhysics.CollisionAPI.Apply(link_prim)

        print(f"Added physics properties to {link_path}")

def setup_joints_for_humanoid(robot_path="/World/HumanoidRobot"):
    """
    Add joints between humanoid robot links
    This is a simplified joint setup - real robots would have more complex joint structures
    """
    stage = omni.usd.get_context().get_stage()

    # Add joints between links
    # Base to Torso (fixed joint for simplicity)
    base_to_torso_path = f"{robot_path}/BaseToTorso"
    joint_prim = stage.DefinePrim(base_to_torso_path, "PhysicsFixedJoint")

    # Torso to Head (ball joint for rotation)
    torso_to_head_path = f"{robot_path}/TorsoToHead"
    joint_prim = stage.DefinePrim(torso_to_head_path, "PhysicsSphericalJoint")

    # Add other joints similarly...
    print("Added basic joints to humanoid robot")

def main():
    """
    Main function to setup humanoid robot in Isaac Sim
    """
    # Create the robot
    create_humanoid_robot_stage()

    # Setup physics
    setup_physics_for_humanoid()

    # Setup joints
    setup_joints_for_humanoid()

    print("Humanoid robot setup complete!")

if __name__ == "__main__":
    main()
```

## 6. Isaac Sim Scene Configuration

### Creating Custom Environments:
```python
# environment_setup.py
import omni
from pxr import Usd, UsdGeom, Gf, Sdf
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage, open_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import define_prim
import numpy as np

def create_humanoid_training_environment():
    """
    Create a training environment suitable for humanoid robots
    """
    stage = omni.usd.get_context().get_stage()

    # Create ground plane
    ground_path = "/World/ground"
    UsdGeom.Xform.Define(stage, "/World")
    ground_prim = UsdGeom.Mesh.Define(stage, ground_path)

    # Set up ground plane geometry
    vertices = [
        Gf.Vec3f(-10, -10, 0), Gf.Vec3f(10, -10, 0),
        Gf.Vec3f(10, 10, 0), Gf.Vec3f(-10, 10, 0)
    ]
    faces = [0, 1, 2, 0, 2, 3]

    ground_prim.CreatePointsAttr(vertices)
    ground_prim.CreateFaceVertexIndicesAttr(faces)
    ground_prim.CreateFaceVertexCountsAttr([3, 3])

    # Add material to ground
    add_ground_material(ground_path)

    # Create obstacles for training
    create_training_obstacles()

    print("Training environment created!")

def add_ground_material(ground_path):
    """
    Add realistic ground material
    """
    stage = omni.usd.get_context().get_stage()

    # Create material
    material_path = "/World/Looks/GroundMaterial"
    material = UsdShade.Material.Define(stage, material_path)

    # Create preview surface shader
    shader_path = f"{material_path}/Shader"
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr("UsdPreviewSurface")

    # Set material properties
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.4, 0.6, 0.3))  # Greenish color
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)

    # Bind material to geometry
    rel = material.CreateSurfaceOutput().GetRawConnectionPathRel()
    rel.SetTargets([shader_path])

def create_training_obstacles():
    """
    Create various obstacles for humanoid robot training
    """
    stage = omni.usd.get_context().get_stage()

    # Create different types of obstacles
    obstacles = [
        {"name": "BoxObstacle1", "position": (2, 0, 0.5), "size": (0.5, 0.5, 1.0)},
        {"name": "CylinderObstacle1", "position": (-2, 1, 0.5), "radius": 0.3, "height": 1.0},
        {"name": "SphereObstacle1", "position": (0, -2, 0.5), "radius": 0.4},
        {"name": "Ramp1", "position": (3, -1, 0), "rotation": (0, 0, 15)},  # 15 degree incline
    ]

    for i, obs in enumerate(obstacles):
        if "size" in obs:
            # Create box obstacle
            box_path = f"/World/Obstacles/{obs['name']}"
            box_geom = UsdGeom.Cube.Define(stage, box_path)
            box_geom.GetSizeAttr().Set(obs['size'][0])
            translate_op = box_geom.AddTranslateOp()
            translate_op.Set(Gf.Vec3f(*obs['position']))

        elif "radius" in obs and "height" in obs:
            # Create cylinder obstacle
            cyl_path = f"/World/Obstacles/{obs['name']}"
            cyl_geom = UsdGeom.Cylinder.Define(stage, cyl_path)
            cyl_geom.GetRadiusAttr().Set(obs['radius'])
            cyl_geom.GetHeightAttr().Set(obs['height'])
            translate_op = cyl_geom.AddTranslateOp()
            translate_op.Set(Gf.Vec3f(*obs['position']))

        elif "radius" in obs:
            # Create sphere obstacle
            sph_path = f"/World/Obstacles/{obs['name']}"
            sph_geom = UsdGeom.Sphere.Define(stage, sph_path)
            sph_geom.GetRadiusAttr().Set(obs['radius'])
            translate_op = sph_geom.AddTranslateOp()
            translate_op.Set(Gf.Vec3f(*obs['position']))

    print(f"Created {len(obstacles)} training obstacles")

def configure_advanced_rendering():
    """
    Configure Isaac Sim for advanced rendering and photorealism
    """
    # Access rendering settings
    carb.settings.get_settings().set("/rtx/transient/domeLightCacheEnabled", False)
    carb.settings.get_settings().set("/rtx/pathTracing/enabled", True)
    carb.settings.get_settings().set("/rtx/pathTracing/maxBounces", 8)

    # Enable advanced rendering features
    carb.settings.get_settings().set("/rtx/ambientOcclusion/enabled", True)
    carb.settings.get_settings().set("/rtx/indirectDiffuse/enabled", True)
    carb.settings.get_settings().set("/rtx/directLighting/enable", True)

    print("Advanced rendering configured")

def setup_lighting_environment():
    """
    Set up professional lighting for photorealistic rendering
    """
    stage = omni.usd.get_context().get_stage()

    # Add dome light (environment light)
    dome_light_path = "/World/DomeLight"
    dome_light = UsdGeom.Sphere.Define(stage, dome_light_path)
    # Actually, let's use the proper dome light prim
    from omni.kit.widget.stage_proxies.model_creation import create_primitive
    # We'll use the proper Omni light primitives
    carb.settings.get_settings().set("/app/stage/updateSelection", True)

    # For now, let's create a strong directional light
    sun_light_path = "/World/SunLight"
    from omni import light
    # Use USD schema for lights
    from pxr import UsdLux
    dome_light_prim = UsdLux.DomeLight.Define(stage, sun_light_path)
    dome_light_prim.CreateIntensityAttr(1000)  # Strong intensity
    dome_light_prim.CreateColorAttr(Gf.Vec3f(1, 1, 1))  # White light

    print("Lighting environment configured")
```

## 7. Synthetic Data Generation Setup

### Configuring Synthetic Data Generation:
```python
# synthetic_data_setup.py
import omni
from omni.isaac.synthetic_utils import SyntheticDataHelper
from omni.isaac.core import World
from omni.replicator.core import Replicator, random_colours
import omni.replicator.core as rep
import numpy as np
import cv2

class SyntheticDataManager:
    def __init__(self):
        self.replicator = Replicator()
        self.sd_helper = SyntheticDataHelper()
        self.data_dir = "./synthetic_data"

    def setup_replication_graph(self):
        """
        Setup replication graph for synthetic data generation
        """
        # Create a trigger that runs every frame
        with self.replicator.new_behavior("on_time", interval=1) as trigger:
            with trigger:
                # Randomize lighting
                self.randomize_lighting()

                # Randomize object poses
                self.randomize_object_poses()

                # Randomize materials
                self.randomize_materials()

    def randomize_lighting(self):
        """
        Randomize lighting conditions for synthetic data
        """
        # Randomize dome light intensity and color
        with rep.randomizer:
            dome_light = rep.get.light(path="/World/DomeLight")
            with dome_light:
                rep.modify.visibility(cone_aperture=rep.distribution.uniform(0.5, 2.0))
                rep.modify.intensity(rep.distribution.uniform(500, 2000))
                rep.modify.color(rep.distribution.uniform((0.8, 0.8, 0.8), (1.2, 1.2, 1.2)))

    def randomize_object_poses(self):
        """
        Randomize object positions and orientations
        """
        # Randomize robot position
        with rep.randomizer:
            robot = rep.get.prim_at_path("/World/HumanoidRobot")
            with robot:
                rep.modify.pose(
                    position=rep.distribution.uniform((-2, -2, 0), (2, 2, 0.5)),
                    rotation=rep.distribution.uniform((-10, -10, -180), (10, 10, 180))
                )

    def randomize_materials(self):
        """
        Randomize materials for domain randomization
        """
        # Create random materials
        with rep.randomizer:
            prims = rep.get.prims(path_pattern="/World/HumanoidRobot/*")
            with prims:
                rep.randomize.material(
                    # Add random materials
                )

    def setup_cameras_for_data_capture(self):
        """
        Setup multiple cameras for comprehensive data capture
        """
        # Create multiple camera viewpoints
        camera_configs = [
            {"name": "front_view", "position": (3, 0, 1.5), "target": (0, 0, 0.8)},
            {"name": "side_view", "position": (0, 3, 1.5), "target": (0, 0, 0.8)},
            {"name": "top_down", "position": (0, 0, 3), "target": (0, 0, 0.8)},
            {"name": "close_up", "position": (1, 1, 1), "target": (0, 0, 0.8)}
        ]

        for cam_config in camera_configs:
            self.create_camera(**cam_config)

    def create_camera(self, name, position, target):
        """
        Create a camera with specified position and target
        """
        from pxr import UsdGeom, Gf
        stage = omni.usd.get_context().get_stage()

        camera_path = f"/World/Cameras/{name}"
        camera_prim = UsdGeom.Camera.Define(stage, camera_path)

        # Set camera properties
        camera_prim.GetFocalLengthAttr().Set(24.0)
        camera_prim.GetHorizontalApertureAttr().Set(36.0)
        camera_prim.GetVerticalApertureAttr().Set(20.25)

        # Position the camera
        xform = UsdGeom.Xformable(camera_prim)
        translate_op = xform.AddTranslateOp()
        translate_op.Set(Gf.Vec3f(*position))

        # Look at target
        # Note: Actual look-at implementation would require more complex transforms
        print(f"Created camera {name} at {position} looking at {target}")

    def setup_data_annotations(self):
        """
        Setup annotations for synthetic data
        """
        # Register annotators for different data types
        self.replicator.register_sensor("/World/Cameras/front_view", "rgb")
        self.replicator.register_sensor("/World/Cameras/front_view", "depth")
        self.replicator.register_sensor("/World/Cameras/front_view", "instance_segmentation")
        self.replicator.register_sensor("/World/Cameras/front_view", "bounding_box_2d_tight")

        print("Data annotation setup complete")

    def generate_synthetic_dataset(self, num_samples=1000):
        """
        Generate a synthetic dataset
        """
        print(f"Generating {num_samples} synthetic data samples...")

        # Initialize replicator
        self.setup_replication_graph()
        self.setup_cameras_for_data_capture()
        self.setup_data_annotations()

        # Start the replicator
        self.replicator.setup_writer(f"{self.data_dir}/writer")
        self.replicator.run(num_samples)

        print(f"Synthetic dataset generated in {self.data_dir}")

def setup_synthetic_data_pipeline():
    """
    Complete setup for synthetic data generation pipeline
    """
    data_manager = SyntheticDataManager()
    data_manager.generate_synthetic_dataset(num_samples=100)

    print("Synthetic data pipeline setup complete!")
```

## 8. Isaac Sim Integration with ROS

### ROS Bridge Configuration:
```python
# isaac_ros_bridge.py
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.extensions import enable_extension
import rospy
from sensor_msgs.msg import JointState, Image, CameraInfo
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import Header
import numpy as np
import cv2
from cv_bridge import CvBridge

class IsaacROSBridge:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('isaac_sim_ros_bridge', anonymous=True)

        # Initialize bridge
        self.bridge = CvBridge()
        self.world = World()

        # Publishers
        self.joint_state_pub = rospy.Publisher('/joint_states', JointState, queue_size=10)
        self.camera_image_pub = rospy.Publisher('/camera/rgb/image_raw', Image, queue_size=10)
        self.camera_info_pub = rospy.Publisher('/camera/rgb/camera_info', CameraInfo, queue_size=10)

        # Subscribers
        self.cmd_vel_sub = rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_callback)

        # Robot state
        self.joint_positions = {}
        self.joint_velocities = {}
        self.joint_efforts = {}

        print("Isaac Sim ROS Bridge initialized")

    def cmd_vel_callback(self, msg):
        """
        Handle velocity commands from ROS
        """
        # Convert ROS twist command to Isaac Sim robot control
        # This would typically interface with the robot's control system
        linear_vel = msg.linear
        angular_vel = msg.angular

        # Apply control to simulated robot
        self.apply_robot_control(linear_vel, angular_vel)

    def apply_robot_control(self, linear_vel, angular_vel):
        """
        Apply control commands to the simulated robot
        """
        # This is a simplified example - actual implementation would depend on robot model
        # and control interface
        print(f"Applying control - Linear: ({linear_vel.x}, {linear_vel.y}, {linear_vel.z}), "
              f"Angular: ({angular_vel.x}, {angular_vel.y}, {angular_vel.z})")

    def publish_joint_states(self):
        """
        Publish joint state information to ROS
        """
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base_link"

        # Populate joint state data
        # This would come from the Isaac Sim robot state
        msg.name = list(self.joint_positions.keys())
        msg.position = list(self.joint_positions.values())
        msg.velocity = list(self.joint_velocities.values())
        msg.effort = list(self.joint_efforts.values())

        self.joint_state_pub.publish(msg)

    def publish_camera_data(self, camera_data):
        """
        Publish camera data to ROS
        """
        # Convert Isaac Sim camera data to ROS Image message
        img_msg = self.bridge.cv2_to_imgmsg(camera_data, encoding="bgr8")
        img_msg.header.stamp = rospy.Time.now()
        img_msg.header.frame_id = "camera_rgb_optical_frame"

        self.camera_image_pub.publish(img_msg)

        # Publish camera info
        cam_info = CameraInfo()
        cam_info.header = img_msg.header
        cam_info.width = camera_data.shape[1]
        cam_info.height = camera_data.shape[0]
        # Add intrinsic parameters
        cam_info.K = [554.256, 0.0, 320.0, 0.0, 554.256, 240.0, 0.0, 0.0, 1.0]  # Example values
        cam_info.D = [0.0, 0.0, 0.0, 0.0, 0.0]  # Distortion coefficients
        cam_info.distortion_model = "plumb_bob"

        self.camera_info_pub.publish(cam_info)

    def run_bridge(self):
        """
        Main bridge loop
        """
        rate = rospy.Rate(60)  # 60 Hz

        while not rospy.is_shutdown():
            # Step Isaac Sim world
            self.world.step(render=True)

            # Publish robot state
            self.publish_joint_states()

            # Publish sensor data (when available)
            # self.publish_camera_data(camera_frame)

            rate.sleep()

def main():
    """
    Main function to run Isaac Sim ROS bridge
    """
    # Enable required extensions
    enable_extension("omni.isaac.ros_bridge")

    # Initialize bridge
    ros_bridge = IsaacROSBridge()

    try:
        ros_bridge.run_bridge()
    except rospy.ROSInterruptException:
        print("ROS Interrupt received, shutting down Isaac Sim bridge")
    except KeyboardInterrupt:
        print("Keyboard interrupt received, shutting down Isaac Sim bridge")

if __name__ == "__main__":
    main()
```

## 9. Performance Optimization for Isaac Sim

### Optimizing Isaac Sim Performance:
```python
# performance_optimizer.py
import carb
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.settings import disable_all_extensions, enable_extension

class IsaacPerformanceOptimizer:
    def __init__(self):
        self.settings = carb.settings.get_settings()

    def optimize_for_training(self):
        """
        Optimize Isaac Sim settings for training scenarios
        """
        # Disable expensive rendering features during training
        self.settings.set("/rtx/pathTracing/enabled", False)
        self.settings.set("/rtx/denoiser/enableDenoising", False)
        self.settings.set("/rtx/transient/dlgiCacheEnabled", False)
        self.settings.set("/rtx/transient/domeLightCacheEnabled", False)

        # Reduce physics accuracy for faster simulation
        self.settings.set("/physics/timeStepsPerSecond", 60)  # Lower for training speed

        # Disable unnecessary features
        self.settings.set("/app/showViewports", False)
        self.settings.set("/persistent/app/viewport/displayOptions", 0)

        print("Optimized for training performance")

    def optimize_for_visualization(self):
        """
        Optimize Isaac Sim settings for visualization quality
        """
        # Enable high-quality rendering features
        self.settings.set("/rtx/pathTracing/enabled", True)
        self.settings.set("/rtx/denoiser/enableDenoising", True)
        self.settings.set("/rtx/transient/dlgiCacheEnabled", True)
        self.settings.set("/rtx/transient/domeLightCacheEnabled", True)

        # Increase physics accuracy
        self.settings.set("/physics/timeStepsPerSecond", 600)  # Higher for accuracy

        # Enable viewport display
        self.settings.set("/app/showViewports", True)

        print("Optimized for visualization quality")

    def optimize_for_synthetic_data(self):
        """
        Optimize Isaac Sim settings for synthetic data generation
        """
        # Balance between quality and performance
        self.settings.set("/rtx/pathTracing/enabled", True)
        self.settings.set("/rtx/denoiser/enableDenoising", False)  # Disable for speed

        # Appropriate physics settings
        self.settings.set("/physics/timeStepsPerSecond", 240)

        # Enable necessary features for data generation
        self.settings.set("/app/showViewports", True)

        print("Optimized for synthetic data generation")

    def set_render_resolution(self, width=1920, height=1080):
        """
        Set render resolution for optimal performance
        """
        self.settings.set("/app/window/resolution", [width, height])
        print(f"Set render resolution to {width}x{height}")

    def manage_texture_streaming(self, enabled=True):
        """
        Manage texture streaming for memory optimization
        """
        self.settings.set("/renderer/textureStreaming/enabled", enabled)
        print(f"Texture streaming {'enabled' if enabled else 'disabled'}")

    def optimize_physics_for_humanoid(self):
        """
        Optimize physics specifically for humanoid robot simulation
        """
        # Set appropriate solver settings for humanoid dynamics
        self.settings.set("/physics/solverType", "TGS")  # TGS solver often better for robotics
        self.settings.set("/physics/frictionModel", "CoulombFriction")  # Realistic friction
        self.settings.set("/physics/enableCCD", True)  # Enable CCD for thin humanoid limbs

        # Set appropriate gravity for Earth-like simulation
        self.settings.set("/physics/gravity", [-9.81, 0.0, 0.0])

        print("Optimized physics for humanoid robot simulation")

def apply_performance_settings(mode="training"):
    """
    Apply appropriate performance settings based on mode
    """
    optimizer = IsaacPerformanceOptimizer()

    if mode == "training":
        optimizer.optimize_for_training()
        optimizer.set_render_resolution(640, 480)  # Lower res for training speed
    elif mode == "visualization":
        optimizer.optimize_for_visualization()
        optimizer.set_render_resolution(1920, 1080)  # Full HD for quality
    elif mode == "synthetic_data":
        optimizer.optimize_for_synthetic_data()
        optimizer.set_render_resolution(1280, 720)  # HD for good quality/speed balance

    optimizer.optimize_physics_for_humanoid()
    optimizer.manage_texture_streaming(enabled=(mode != "training"))

    print(f"Performance settings applied for {mode} mode")
```

## 10. Troubleshooting Isaac Sim Installation

### Common Issues and Solutions:

1. **GPU Memory Issues**:
   ```bash
   # Check GPU memory usage
   nvidia-smi

   # Reduce scene complexity or increase swap space
   # Optimize USD assets for lower memory usage
   ```

2. **Driver Compatibility Issues**:
   ```bash
   # Ensure latest NVIDIA drivers are installed
   sudo apt install nvidia-driver-535  # Or latest version
   # Restart system after driver installation
   ```

3. **Omniverse Connection Issues**:
   ```bash
   # Check Omniverse services
   systemctl status omni.services
   # Verify network connectivity
   ping localhost
   ```

4. **Python Environment Issues**:
   ```bash
   # Use Isaac Sim's Python environment
   ~/.local/share/ov/pkg/isaac_sim-[VERSION]/python.sh -c "import omni; print('Success')"
   ```

### Isaac Sim Diagnostic Tool:
```python
# isaac_sim_diagnostics.py
import sys
import subprocess
import platform
import omni
from omni.isaac.core import World
import carb

def run_isaac_sim_diagnostics():
    """
    Run comprehensive diagnostics for Isaac Sim installation
    """
    print("=== Isaac Sim Diagnostics ===")
    print(f"Platform: {platform.platform()}")
    print(f"Python version: {sys.version}")

    # Check GPU
    try:
        gpu_info = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], universal_newlines=True)
        print(f"GPU: {gpu_info.strip()}")
    except:
        print("ERROR: Could not detect NVIDIA GPU")

    # Check Isaac Sim modules
    try:
        import omni
        print("✓ omni module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import omni: {e}")

    try:
        import omni.isaac.core
        print("✓ omni.isaac.core imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import omni.isaac.core: {e}")

    try:
        import omni.replicator.core
        print("✓ omni.replicator.core imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import omni.replicator.core: {e}")

    # Check stage access
    try:
        stage = omni.usd.get_context().get_stage()
        if stage:
            print("✓ Stage access confirmed")
        else:
            print("✗ Could not access stage")
    except Exception as e:
        print(f"✗ Stage access error: {e}")

    # Check settings
    try:
        settings = carb.settings.get_settings()
        print("✓ Settings access confirmed")
    except Exception as e:
        print(f"✗ Settings access error: {e}")

    print("\n=== Diagnostics Complete ===")

if __name__ == "__main__":
    run_isaac_sim_diagnostics()
```

## 11. Best Practices for Isaac Sim Setup

### Environment Setup Best Practices:
- **Start simple**: Begin with basic scenes and gradually increase complexity
- **Use appropriate resolutions**: Match render resolution to your use case
- **Manage assets efficiently**: Organize assets in a logical hierarchy
- **Version control**: Use USD stage versioning for experiment reproducibility
- **Backup configurations**: Save working configurations for quick restoration

### Performance Best Practices:
- **Use proxy shapes**: During scene construction, use proxy representations
- **Instance geometry**: Reuse identical assets through instancing
- **Optimize materials**: Use efficient material networks
- **Cull unnecessary geometry**: Hide objects not in the camera view
- **Use Level of Detail (LOD)**: Implement LOD systems for complex scenes

### Data Generation Best Practices:
- **Domain randomization**: Vary lighting, textures, and object positions
- **Annotation accuracy**: Ensure high-quality ground truth data
- **Consistent naming**: Use consistent naming conventions for assets
- **Metadata logging**: Log all randomization parameters for reproducibility
- **Quality checks**: Implement validation pipelines for generated data

## Weekly Schedule Focus (Weeks 8-10)
During Weeks 8-10, we will focus on:
- NVIDIA Isaac Sim: Photorealistic simulation and synthetic data generation
- Isaac ROS: Hardware-accelerated VSLAM (Visual SLAM) and navigation
- Nav2: Path planning for bipedal humanoid movement
- Advanced perception and training systems

## Resources
- [NVIDIA Isaac Sim Documentation](https://docs.omniverse.nvidia.com/isaacsim/latest/index.html)
- [Isaac Sim Tutorials](https://docs.omniverse.nvidia.com/isaacsim/latest/tutorial_intro.html)
- [Omniverse Nucleus Setup](https://docs.omniverse.nvidia.com/nucleus/latest/index.html)
- [ROS Bridge for Isaac Sim](https://docs.omniverse.nvidia.com/isaacsim/latest/tutorial_ros.html)
- [Synthetic Data Generation Guide](https://docs.omniverse.nvidia.com/isaacsim/latest/features/synthetic_data_generation.html)
