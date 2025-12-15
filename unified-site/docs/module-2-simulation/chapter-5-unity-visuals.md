# Chapter 5: Unity Visuals for Robot Visualization

## Overview
This chapter covers the use of Unity for high-fidelity robot visualization and human-robot interaction in simulation environments. Students will learn to create photorealistic visualizations of humanoid robots, develop interactive interfaces, and understand the integration between Unity and robotics frameworks for enhanced simulation experiences.

## Learning Objectives
By the end of this chapter, students will be able to:
- Set up Unity for robotics visualization applications
- Import and configure robot models in Unity
- Create interactive visualization environments
- Implement real-time robot control visualization
- Understand Unity's role in the digital twin ecosystem
- Integrate Unity with ROS 2 for bidirectional communication

## 1. Introduction to Unity for Robotics

Unity is a powerful game engine that provides high-fidelity visualization capabilities for robotics applications. For humanoid robots, Unity offers:

- **Photorealistic rendering**: Advanced lighting, materials, and textures
- **Real-time physics**: Interactive simulation with realistic physics
- **Cross-platform deployment**: Run on various hardware configurations
- **Rich interaction systems**: VR/AR support and human-robot interaction
- **Asset ecosystem**: Extensive library of models, materials, and tools

### Unity vs Gazebo for Visualization:
- **Unity**: Focus on visual quality and human interaction
- **Gazebo**: Focus on physics accuracy and sensor simulation
- **Best practice**: Use both together for complete simulation pipeline

## 2. Setting Up Unity for Robotics

### Unity Installation Requirements:
- Unity Hub (recommended for version management)
- Unity 2022.3 LTS or later (for stability and support)
- Graphics card with DirectX 11 or OpenGL 4.3 support
- VRAM: 4GB+ recommended for complex robot models

### Unity Robotics Packages:
```bash
# Install Unity Robotics packages via Unity Package Manager
# In Unity Editor: Window > Package Manager
# Add package by name or git URL:

# ROS-TCP-Connector: For ROS 2 communication
com.unity.robotics.ros-tcp-connector

# URDF-Importer: For importing robot models
com.unity.robotics.urdf-importer

# Simulation-Controls: For simulation management
com.unity.robotics.simulation-controls
```

### Basic Unity Project Setup for Robotics:
1. Create new 3D project in Unity Hub
2. Install required robotics packages
3. Configure build settings for your target platform
4. Set up physics settings appropriate for robotics

## 3. Importing Robot Models with URDF Importer

### Using URDF Importer Package:
```csharp
// Example: Importing a URDF robot programmatically
using Unity.Robotics.URDFImporter;
using UnityEngine;

public class RobotImporter : MonoBehaviour
{
    [Header("URDF Settings")]
    public string urdfPath;
    public JointControlType defaultJointType = JointControlType.Physics;

    void Start()
    {
        if (!string.IsNullOrEmpty(urdfPath))
        {
            // Import robot from URDF file
            var robot = URDFRobotExtensions.CreateRobot(urdfPath, defaultJointType);
            if (robot != null)
            {
                robot.transform.SetParent(transform);
                Debug.Log($"Successfully imported robot from: {urdfPath}");
            }
        }
    }
}
```

### Preparing URDF Files for Unity:
```xml
<!-- Unity-friendly URDF example -->
<?xml version="1.0"?>
<robot name="humanoid_robot">
  <!-- Links with Unity-compatible materials -->
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
      <material name="base_material">
        <color rgba="0.8 0.8 0.8 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
    </collision>
  </link>

  <!-- Unity-specific visual enhancements -->
  <gazebo reference="base_link">
    <material>Gazebo/Grey</material>
  </gazebo>
</robot>
```

### Material and Texture Setup:
```csharp
// Unity script for setting up robot materials
using UnityEngine;
using Unity.Robotics.ROSTCPConnector.ROSGeometry;
using System.Collections.Generic;

public class RobotMaterialSetup : MonoBehaviour
{
    [Header("Material Settings")]
    public Material robotBodyMaterial;
    public Material jointMaterial;
    public Material sensorMaterial;

    void Start()
    {
        SetupRobotMaterials();
    }

    void SetupRobotMaterials()
    {
        // Find all robot parts and assign appropriate materials
        var links = GetComponentsInChildren<Renderer>();

        foreach (var link in links)
        {
            if (link.name.Contains("joint"))
            {
                link.material = jointMaterial;
            }
            else if (link.name.Contains("sensor"))
            {
                link.material = sensorMaterial;
            }
            else
            {
                link.material = robotBodyMaterial;
            }
        }
    }
}
```

## 4. Creating Interactive Visualization Environments

### Environment Setup Script:
```csharp
// EnvironmentController.cs
using UnityEngine;
using UnityEngine.Rendering;
using System.Collections;

public class EnvironmentController : MonoBehaviour
{
    [Header("Environment Settings")]
    public Light mainLight;
    public Material groundMaterial;
    public GameObject[] interactiveElements;

    [Header("Camera Settings")]
    public Camera mainCamera;
    public float cameraMoveSpeed = 5.0f;
    public float cameraRotateSpeed = 2.0f;

    [Header("Robot Control")]
    public GameObject robot;

    void Start()
    {
        SetupEnvironment();
    }

    void SetupEnvironment()
    {
        // Configure lighting
        if (mainLight != null)
        {
            mainLight.shadows = LightShadows.Soft;
            RenderSettings.ambientLight = Color.grey;
        }

        // Create ground plane
        CreateGroundPlane();

        // Setup interactive elements
        foreach (var element in interactiveElements)
        {
            SetupInteractiveElement(element);
        }
    }

    void CreateGroundPlane()
    {
        var ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
        ground.name = "Ground";
        ground.transform.position = Vector3.zero;
        ground.transform.localScale = new Vector3(10, 1, 10);

        if (groundMaterial != null)
        {
            ground.GetComponent<Renderer>().material = groundMaterial;
        }
    }

    void SetupInteractiveElement(GameObject element)
    {
        // Add interaction components
        var rb = element.AddComponent<Rigidbody>();
        rb.isKinematic = true; // Don't let physics control it

        // Add collider for interaction detection
        if (element.GetComponent<Collider>() == null)
        {
            element.AddComponent<BoxCollider>();
        }
    }

    void Update()
    {
        HandleCameraControls();
        UpdateEnvironment();
    }

    void HandleCameraControls()
    {
        // Camera movement controls
        if (Input.GetKey(KeyCode.W))
            mainCamera.transform.position += mainCamera.transform.forward * cameraMoveSpeed * Time.deltaTime;
        if (Input.GetKey(KeyCode.S))
            mainCamera.transform.position -= mainCamera.transform.forward * cameraMoveSpeed * Time.deltaTime;
        if (Input.GetKey(KeyCode.A))
            mainCamera.transform.position -= mainCamera.transform.right * cameraMoveSpeed * Time.deltaTime;
        if (Input.GetKey(KeyCode.D))
            mainCamera.transform.position += mainCamera.transform.right * cameraMoveSpeed * Time.deltaTime;

        // Camera rotation
        if (Input.GetMouseButton(1)) // Right mouse button
        {
            float mouseX = Input.GetAxis("Mouse X") * cameraRotateSpeed;
            float mouseY = Input.GetAxis("Mouse Y") * cameraRotateSpeed;

            mainCamera.transform.Rotate(-mouseY, mouseX, 0);
        }
    }

    void UpdateEnvironment()
    {
        // Update environment-specific elements
        // This could include dynamic lighting, particle effects, etc.
    }
}
```

### Interactive Robot Control Interface:
```csharp
// RobotControlInterface.cs
using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;

public class RobotControlInterface : MonoBehaviour
{
    [Header("UI Elements")]
    public Slider[] jointSliders;
    public Text[] jointValueTexts;
    public Button resetButton;
    public Toggle simulationToggle;

    [Header("Robot Reference")]
    public GameObject robot;

    private Dictionary<string, ArticulationBody> jointMap;
    private List<ArticulationBody> joints;

    void Start()
    {
        InitializeRobotControls();
        SetupUIEvents();
    }

    void InitializeRobotControls()
    {
        // Find all articulation bodies in the robot
        joints = new List<ArticulationBody>();
        jointMap = new Dictionary<string, ArticulationBody>();

        var articulationBodies = robot.GetComponentsInChildren<ArticulationBody>();
        foreach (var body in articulationBodies)
        {
            joints.Add(body);
            jointMap[body.name] = body;

            // Create UI elements for each joint
            if (joints.Count <= jointSliders.Length)
            {
                var joint = body;
                var slider = jointSliders[joints.Count - 1];

                // Configure slider based on joint limits
                var drive = joint.xDrive;
                slider.minValue = drive.lowerLimit;
                slider.maxValue = drive.upperLimit;
                slider.value = 0; // Default position
            }
        }
    }

    void SetupUIEvents()
    {
        // Setup slider value changed events
        for (int i = 0; i < jointSliders.Length && i < joints.Count; i++)
        {
            int index = i; // Capture for closure
            jointSliders[i].onValueChanged.AddListener((value) =>
            {
                UpdateJointPosition(index, value);
            });
        }

        // Setup reset button
        if (resetButton != null)
        {
            resetButton.onClick.AddListener(ResetRobot);
        }
    }

    void UpdateJointPosition(int jointIndex, float position)
    {
        if (jointIndex < joints.Count)
        {
            var joint = joints[jointIndex];
            var drive = joint.xDrive;
            drive.target = position;
            joint.xDrive = drive;

            // Update UI text
            if (jointIndex < jointValueTexts.Length)
            {
                jointValueTexts[jointIndex].text = $"Joint {jointIndex + 1}: {position:F2}";
            }
        }
    }

    void ResetRobot()
    {
        // Reset all joint positions to default
        for (int i = 0; i < jointSliders.Length && i < joints.Count; i++)
        {
            jointSliders[i].value = 0;
            UpdateJointPosition(i, 0);
        }
    }
}
```

## 5. ROS 2 Integration with Unity

### ROS-TCP-Connector Setup:
```csharp
// ROSCommunicationManager.cs
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageGeneration;
using System.Collections.Generic;

public class ROSCommunicationManager : MonoBehaviour
{
    [Header("ROS Connection")]
    public string rosIP = "127.0.0.1";
    public int rosPort = 10000;

    [Header("Robot Topics")]
    public string jointStateTopic = "/joint_states";
    public string robotCommandTopic = "/joint_commands";

    private ROSConnection ros;
    private Dictionary<string, float[]> jointPositions;
    private Dictionary<string, float[]> jointVelocities;

    void Start()
    {
        // Initialize ROS connection
        ros = ROSConnection.GetOrCreateInstance();
        ros.Initialize(rosIP, rosPort);

        // Initialize joint dictionaries
        jointPositions = new Dictionary<string, float[]>();
        jointVelocities = new Dictionary<string, float[]>();

        // Subscribe to ROS topics
        ros.Subscribe<sensor_msgs.msg.JointState>(jointStateTopic, OnJointStateReceived);

        Debug.Log($"ROS Communication Manager initialized. Connecting to {rosIP}:{rosPort}");
    }

    void OnJointStateReceived(sensor_msgs.msg.JointState jointState)
    {
        // Process received joint state message
        jointPositions.Clear();
        jointVelocities.Clear();

        // Store position data
        if (jointState.name.Count == jointState.position.Count)
        {
            for (int i = 0; i < jointState.name.Count; i++)
            {
                if (!jointPositions.ContainsKey(jointState.name[i]))
                {
                    jointPositions.Add(jointState.name[i], new float[1]);
                }
                jointPositions[jointState.name[i]][0] = (float)jointState.position[i];
            }
        }

        // Store velocity data
        if (jointState.name.Count == jointState.velocity.Count)
        {
            for (int i = 0; i < jointState.name.Count; i++)
            {
                if (!jointVelocities.ContainsKey(jointState.name[i]))
                {
                    jointVelocities.Add(jointState.name[i], new float[1]);
                }
                jointVelocities[jointState.name[i]][0] = (float)jointState.velocity[i];
            }
        }

        // Update Unity robot visualization
        UpdateRobotVisualization();
    }

    void UpdateRobotVisualization()
    {
        // Update robot joint positions in Unity
        var robotJoints = GetComponentsInChildren<ArticulationBody>();
        foreach (var joint in robotJoints)
        {
            if (jointPositions.ContainsKey(joint.name))
            {
                var drive = joint.xDrive;
                drive.target = jointPositions[joint.name][0];
                joint.xDrive = drive;
            }
        }
    }

    public void SendJointCommand(Dictionary<string, float> commands)
    {
        // Create and send joint command message
        var jointCmd = new trajectory_msgs.msg.JointTrajectory();
        jointCmd.header.stamp = new builtin_interfaces.msg.Time();
        jointCmd.header.frame_id = "base_link";

        // Set joint names
        foreach (var cmd in commands)
        {
            jointCmd.joint_names.Add(cmd.Key);
        }

        // Create trajectory point
        var point = new trajectory_msgs.msg.JointTrajectoryPoint();
        foreach (var cmd in commands)
        {
            point.positions.Add(cmd.Value);
            point.velocities.Add(0.0f); // Default velocity
            point.accelerations.Add(0.0f); // Default acceleration
        }
        point.time_from_start = new builtin_interfaces.msg.Duration { sec = 1, nanosec = 0 };

        jointCmd.points.Add(point);

        // Publish command
        ros.Publish(robotCommandTopic, jointCmd);
    }

    void OnApplicationQuit()
    {
        if (ros != null)
        {
            ros.Close();
        }
    }
}
```

### Sensor Data Visualization:
```csharp
// SensorVisualization.cs
using UnityEngine;
using System.Collections.Generic;

public class SensorVisualization : MonoBehaviour
{
    [Header("Sensor Visualization")]
    public GameObject lidarVisualizationPrefab;
    public GameObject cameraFrustumPrefab;
    public GameObject imuIndicatorPrefab;

    [Header("Visualization Settings")]
    public float lidarMaxRange = 30.0f;
    public Color lidarColor = Color.red;
    public float visualizationUpdateRate = 0.1f;

    private Dictionary<string, GameObject> sensorVisualizations;
    private float lastUpdate;

    void Start()
    {
        sensorVisualizations = new Dictionary<string, GameObject>();
        lastUpdate = Time.time;
    }

    void Update()
    {
        if (Time.time - lastUpdate > visualizationUpdateRate)
        {
            UpdateSensorVisualizations();
            lastUpdate = Time.time;
        }
    }

    public void AddLidarVisualization(string sensorName, float[] ranges, float[] angles)
    {
        GameObject lidarVis;
        if (!sensorVisualizations.ContainsKey(sensorName))
        {
            // Create new lidar visualization
            lidarVis = Instantiate(lidarVisualizationPrefab, transform);
            lidarVis.name = $"LidarVis_{sensorName}";
            sensorVisualizations[sensorName] = lidarVis;
        }
        else
        {
            lidarVis = sensorVisualizations[sensorName];
        }

        // Update lidar visualization with current ranges
        UpdateLidarVisualization(lidarVis, ranges, angles);
    }

    void UpdateLidarVisualization(GameObject lidarVis, float[] ranges, float[] angles)
    {
        // Clear previous visualization
        foreach (Transform child in lidarVis.transform)
        {
            Destroy(child.gameObject);
        }

        // Create new visualization points
        for (int i = 0; i < ranges.Length; i++)
        {
            if (ranges[i] > 0 && ranges[i] <= lidarMaxRange)
            {
                var point = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                point.transform.SetParent(lidarVis.transform);
                point.transform.localScale = Vector3.one * 0.05f;
                point.GetComponent<Renderer>().material.color = lidarColor;

                float x = ranges[i] * Mathf.Cos(angles[i]);
                float y = 0; // Assuming 2D lidar
                float z = ranges[i] * Mathf.Sin(angles[i]);

                point.transform.localPosition = new Vector3(x, y, z);
                Destroy(point.GetComponent<Collider>()); // Remove collider for performance
            }
        }
    }

    public void AddCameraVisualization(string sensorName, Vector3 position, Vector3 direction, float fov)
    {
        GameObject cameraVis;
        if (!sensorVisualizations.ContainsKey(sensorName))
        {
            cameraVis = Instantiate(cameraFrustumPrefab, transform);
            cameraVis.name = $"CameraVis_{sensorName}";
            sensorVisualizations[sensorName] = cameraVis;
        }
        else
        {
            cameraVis = sensorVisualizations[sensorName];
        }

        // Update camera visualization
        cameraVis.transform.position = position;
        cameraVis.transform.forward = direction;
    }
}
```

## 6. High-Fidelity Rendering Techniques

### Advanced Materials for Robot Parts:
```csharp
// RobotMaterialController.cs
using UnityEngine;
using UnityEngine.Rendering;

public class RobotMaterialController : MonoBehaviour
{
    [Header("Material Properties")]
    public Material robotBodyMaterial;
    public Material jointMaterial;
    public Material sensorMaterial;

    [Header("Rendering Settings")]
    public bool enableSSR = true; // Screen Space Reflections
    public bool enableSSAO = true; // Screen Space Ambient Occlusion
    public float metallicValue = 0.8f;
    public float smoothnessValue = 0.6f;

    void Start()
    {
        SetupAdvancedMaterials();
        ConfigureRenderingSettings();
    }

    void SetupAdvancedMaterials()
    {
        if (robotBodyMaterial != null)
        {
            robotBodyMaterial.SetFloat("_Metallic", metallicValue);
            robotBodyMaterial.SetFloat("_Smoothness", smoothnessValue);
        }

        if (jointMaterial != null)
        {
            jointMaterial.SetFloat("_Metallic", 0.9f); // More metallic for joints
            jointMaterial.SetFloat("_Smoothness", 0.7f);
        }

        if (sensorMaterial != null)
        {
            sensorMaterial.SetFloat("_Metallic", 0.3f); // Less metallic for sensors
            sensorMaterial.SetFloat("_Smoothness", 0.4f);
        }
    }

    void ConfigureRenderingSettings()
    {
        // Configure post-processing effects
        ConfigurePostProcessing();

        // Set up lighting for best visualization
        SetupRobotLighting();
    }

    void ConfigurePostProcessing()
    {
        // Configure rendering pipeline settings
        // This would typically involve setting up Universal Render Pipeline or HDRP
        if (enableSSR)
        {
            // Enable screen space reflections in the render pipeline
            Debug.Log("Screen Space Reflections enabled");
        }

        if (enableSSAO)
        {
            // Enable screen space ambient occlusion
            Debug.Log("Screen Space Ambient Occlusion enabled");
        }
    }

    void SetupRobotLighting()
    {
        // Create and configure lights specifically for robot visualization
        var robotLight = new GameObject("RobotLight").AddComponent<Light>();
        robotLight.type = LightType.Spot;
        robotLight.intensity = 2.0f;
        robotLight.spotAngle = 60f;
        robotLight.color = Color.white;
        robotLight.transform.position = transform.position + Vector3.up * 3f + Vector3.back * 2f;
        robotLight.transform.LookAt(transform.position);
    }
}
```

### Realistic Environment Textures:
```csharp
// EnvironmentMaterialController.cs
using UnityEngine;

[ExecuteInEditMode]
public class EnvironmentMaterialController : MonoBehaviour
{
    [Header("Ground Material")]
    public Material groundMaterial;
    public Texture2D groundAlbedo;
    public Texture2D groundNormal;
    public float groundTiling = 1.0f;

    [Header("Environment Materials")]
    public Material[] environmentMaterials;

    void Start()
    {
        ApplyEnvironmentMaterials();
    }

    void ApplyEnvironmentMaterials()
    {
        if (groundMaterial != null)
        {
            groundMaterial.SetTexture("_BaseMap", groundAlbedo);
            groundMaterial.SetTexture("_BumpMap", groundNormal);
            groundMaterial.mainTextureScale = new Vector3(groundTiling, groundTiling, 1);
        }

        foreach (var material in environmentMaterials)
        {
            if (material != null)
            {
                // Apply consistent environment properties
                material.SetFloat("_Metallic", 0.1f);
                material.SetFloat("_Smoothness", 0.3f);
            }
        }
    }
}
```

## 7. Human-Robot Interaction in Unity

### VR/AR Interaction Setup:
```csharp
// VRInteractionController.cs
using UnityEngine;
using UnityEngine.XR;

public class VRInteractionController : MonoBehaviour
{
    [Header("VR Controllers")]
    public GameObject leftController;
    public GameObject rightController;

    [Header("Interaction Settings")]
    public float interactionDistance = 2.0f;
    public LayerMask interactionLayer;

    [Header("Robot Interaction")]
    public GameObject robot;
    public float robotMoveSpeed = 1.0f;

    void Update()
    {
        HandleVRInput();
        UpdateRobotInteraction();
    }

    void HandleVRInput()
    {
        // Handle VR controller input
        if (leftController != null)
        {
            // Process left controller input
            if (IsGripped(XRNode.LeftHand))
            {
                // Handle grip action
                HandleGripAction(leftController);
            }
        }

        if (rightController != null)
        {
            // Process right controller input
            if (IsGripped(XRNode.RightHand))
            {
                // Handle grip action
                HandleGripAction(rightController);
            }
        }
    }

    bool IsGripped(XRNode node)
    {
        // Check if controller is gripped
        InputDevices.GetDeviceAtXRNode(node).TryGetFeatureValue(CommonUsages.gripButton, out bool isGripped);
        return isGripped;
    }

    void HandleGripAction(GameObject controller)
    {
        // Raycast to find interaction target
        RaycastHit hit;
        Vector3 rayDirection = controller.transform.forward;
        Vector3 rayOrigin = controller.transform.position;

        if (Physics.Raycast(rayOrigin, rayDirection, out hit, interactionDistance, interactionLayer))
        {
            // Handle interaction with hit object
            var interactable = hit.collider.GetComponent<InteractableObject>();
            if (interactable != null)
            {
                interactable.Interact();
            }
        }
    }

    void UpdateRobotInteraction()
    {
        // Update robot based on VR interaction
        // This could involve moving the robot, changing its pose, etc.
    }
}

// Base class for interactable objects
public class InteractableObject : MonoBehaviour
{
    public virtual void Interact()
    {
        Debug.Log($"Interacted with {gameObject.name}");
        // Override this method in derived classes
    }

    void OnMouseOver()
    {
        // Visual feedback when looking at object (for non-VR)
        if (Input.GetMouseButtonDown(0))
        {
            Interact();
        }
    }
}
```

### Gesture Recognition for Human-Robot Interaction:
```csharp
// GestureRecognition.cs
using UnityEngine;
using System.Collections.Generic;

public class GestureRecognition : MonoBehaviour
{
    [Header("Gesture Recognition")]
    public float gestureThreshold = 0.1f;
    public float gestureTimeout = 2.0f;

    [Header("Gesture Events")]
    public UnityEngine.Events.UnityEvent onWaveGesture;
    public UnityEngine.Events.UnityEvent onPointGesture;
    public UnityEngine.Events.UnityEvent onStopGesture;

    private List<Vector3> gesturePoints;
    private float gestureStartTime;
    private bool isRecordingGesture;

    void Start()
    {
        gesturePoints = new List<Vector3>();
    }

    void Update()
    {
        UpdateGestureRecognition();
    }

    public void StartRecordingGesture()
    {
        gesturePoints.Clear();
        gestureStartTime = Time.time;
        isRecordingGesture = true;
    }

    public void StopRecordingGesture()
    {
        if (isRecordingGesture)
        {
            AnalyzeGesture();
            isRecordingGesture = false;
        }
    }

    void UpdateGestureRecognition()
    {
        if (isRecordingGesture)
        {
            // Add current hand position to gesture points
            gesturePoints.Add(GetHandPosition());

            // Check for timeout
            if (Time.time - gestureStartTime > gestureTimeout)
            {
                StopRecordingGesture();
            }
        }
    }

    Vector3 GetHandPosition()
    {
        // This would typically get the position from a VR controller or hand tracking
        // For now, we'll use mouse position as a placeholder
        return Camera.main.ScreenToWorldPoint(new Vector3(
            Input.mousePosition.x,
            Input.mousePosition.y,
            Camera.main.nearClipPlane + 1.0f
        ));
    }

    void AnalyzeGesture()
    {
        if (gesturePoints.Count < 3) return; // Need at least 3 points for gesture

        // Analyze gesture pattern
        var gestureType = RecognizeGesture(gesturePoints.ToArray());

        // Trigger appropriate event
        switch (gestureType)
        {
            case GestureType.Wave:
                onWaveGesture?.Invoke();
                break;
            case GestureType.Point:
                onPointGesture?.Invoke();
                break;
            case GestureType.Stop:
                onStopGesture?.Invoke();
                break;
        }
    }

    GestureType RecognizeGesture(Vector3[] points)
    {
        // Simple gesture recognition based on movement patterns
        if (points.Length < 3) return GestureType.None;

        // Calculate movement characteristics
        Vector3 firstPoint = points[0];
        Vector3 lastPoint = points[points.Length - 1];
        Vector3 totalMovement = lastPoint - firstPoint;

        // Wave gesture: back-and-forth horizontal movement
        if (Mathf.Abs(totalMovement.x) > gestureThreshold &&
            Mathf.Abs(totalMovement.y) < gestureThreshold * 0.5f)
        {
            // Check for oscillating pattern
            bool isOscillating = IsOscillatingMovement(points, Vector3.right);
            if (isOscillating) return GestureType.Wave;
        }

        // Point gesture: straight movement in one direction
        if (totalMovement.magnitude > gestureThreshold)
        {
            return GestureType.Point;
        }

        // Stop gesture: minimal movement
        if (totalMovement.magnitude < gestureThreshold * 0.5f)
        {
            return GestureType.Stop;
        }

        return GestureType.None;
    }

    bool IsOscillatingMovement(Vector3[] points, Vector3 direction)
    {
        // Check if movement oscillates along a particular axis
        float lastProjection = Vector3.Dot(points[0], direction);

        int directionChanges = 0;
        for (int i = 1; i < points.Length; i++)
        {
            float currentProjection = Vector3.Dot(points[i], direction);
            float projectionDiff = currentProjection - lastProjection;

            if (projectionDiff * lastProjection < 0) // Sign changed
            {
                directionChanges++;
            }

            lastProjection = currentProjection;
        }

        return directionChanges > 2; // At least 2 direction changes for oscillating
    }
}

public enum GestureType
{
    None,
    Wave,
    Point,
    Stop
}
```

## 8. Performance Optimization for Unity Robotics

### Level of Detail (LOD) System:
```csharp
// RobotLODController.cs
using UnityEngine;

[RequireComponent(typeof(LODGroup))]
public class RobotLODController : MonoBehaviour
{
    [Header("LOD Settings")]
    public float[] lodDistances = { 10f, 30f, 60f };
    public bool enableLOD = true;

    private LODGroup lodGroup;
    private Renderer[] allRenderers;

    void Start()
    {
        SetupLODSystem();
    }

    void SetupLODSystem()
    {
        lodGroup = GetComponent<LODGroup>();
        if (lodGroup == null)
        {
            lodGroup = gameObject.AddComponent<LODGroup>();
        }

        // Get all renderers in the robot hierarchy
        allRenderers = GetComponentsInChildren<Renderer>();

        // Create LOD levels
        LOD[] lods = new LOD[lodDistances.Length + 1]; // +1 for the "no LOD" level

        for (int i = 0; i < lodDistances.Length; i++)
        {
            // Calculate screen relative transition height
            float transitionHeight = lodDistances[i] / Camera.main.farClipPlane;
            lods[i] = new LOD(transitionHeight, GetRenderersForLODLevel(i));
        }

        // Last LOD level (lowest detail)
        lods[lodDistances.Length] = new LOD(0f, GetRenderersForLODLevel(lodDistances.Length));

        lodGroup.SetLODs(lods);
        lodGroup.RecalculateBounds();
    }

    Renderer[] GetRenderersForLODLevel(int level)
    {
        // This is a simplified example - in practice, you'd have different
        // sets of renderers for each LOD level
        System.Collections.Generic.List<Renderer> renderers = new System.Collections.Generic.List<Renderer>();

        for (int i = 0; i < allRenderers.Length; i++)
        {
            // Simple LOD selection based on level
            if (i % (level + 1) == 0)
            {
                renderers.Add(allRenderers[i]);
            }
        }

        return renderers.ToArray();
    }
}
```

### Occlusion Culling for Large Environments:
```csharp
// OcclusionCullingController.cs
using UnityEngine;

public class OcclusionCullingController : MonoBehaviour
{
    [Header("Occlusion Settings")]
    public bool enableOcclusionCulling = true;
    public float cullingUpdateRate = 0.1f;

    private float lastCullingUpdate;

    void Start()
    {
        SetupOcclusionCulling();
    }

    void SetupOcclusionCulling()
    {
        if (enableOcclusionCulling)
        {
            // Unity's occlusion culling is typically set up in the editor
            // This script would manage runtime occlusion updates
            Debug.Log("Occlusion culling enabled");
        }
    }

    void Update()
    {
        if (enableOcclusionCulling && Time.time - lastCullingUpdate > cullingUpdateRate)
        {
            UpdateOcclusion();
            lastCullingUpdate = Time.time;
        }
    }

    void UpdateOcclusion()
    {
        // Update occlusion culling based on current camera position
        // This is handled automatically by Unity's occlusion system
    }
}
```

## 9. Troubleshooting Common Unity Robotics Issues

### Common Issues and Solutions:

1. **URDF Import Failures**:
   - Check that URDF files are properly formatted
   - Verify that mesh files are accessible and in correct format (FBX, OBJ)
   - Ensure Unity has permission to read external files

2. **Performance Issues**:
   - Reduce polygon count of robot models
   - Use LOD system for distant robots
   - Optimize materials and textures
   - Reduce physics update rate if not critical

3. **ROS Connection Problems**:
   - Verify ROS bridge is running
   - Check IP addresses and ports
   - Ensure firewall allows connections
   - Verify ROS message types match Unity expectations

4. **Visual Artifacts**:
   - Check for proper scaling between ROS and Unity units
   - Verify coordinate system conversions (ROS: right-handed, Unity: left-handed)
   - Adjust material properties for proper lighting

### Performance Monitoring:
```csharp
// PerformanceMonitor.cs
using UnityEngine;
using UnityEngine.UI;

public class PerformanceMonitor : MonoBehaviour
{
    [Header("Performance UI")]
    public Text fpsText;
    public Text memoryText;
    public Text drawCallsText;

    [Header("Performance Settings")]
    public float updateInterval = 0.5f;

    private float lastUpdateTime;
    private int frameCount;

    void Start()
    {
        lastUpdateTime = Time.realtimeSinceStartup;
        frameCount = 0;
    }

    void Update()
    {
        frameCount++;
        float currentTime = Time.realtimeSinceStartup;

        if (currentTime - lastUpdateTime >= updateInterval)
        {
            float fps = frameCount / (currentTime - lastUpdateTime);
            frameCount = 0;
            lastUpdateTime = currentTime;

            UpdatePerformanceUI(fps);
        }
    }

    void UpdatePerformanceUI(float fps)
    {
        if (fpsText != null)
            fpsText.text = $"FPS: {fps:F1}";

        if (memoryText != null)
            memoryText.text = $"Memory: {System.GC.GetTotalMemory(false) / 1048576:F1} MB";

        if (drawCallsText != null)
            drawCallsText.text = $"Draw Calls: {UnityEngine.Rendering.UnityRenderPipeline.beginFrameRenderingCount}";
    }
}
```

## 10. Best Practices for Unity Robotics

### Visualization Best Practices:
- Use appropriate scaling (1 Unity unit = 1 meter is recommended)
- Maintain consistent coordinate systems between ROS and Unity
- Use efficient rendering techniques for real-time performance
- Implement proper level of detail for complex scenes

### Integration Best Practices:
- Keep ROS communication on separate threads to avoid blocking Unity
- Use appropriate message rates to balance performance and responsiveness
- Implement proper error handling for connection failures
- Validate data before applying to Unity objects

### Development Workflow:
- Use version control for both Unity project and robot models
- Document coordinate system conventions
- Create modular components for reusability
- Test with various robot configurations

## Weekly Schedule Focus (Weeks 6-7)
During Weeks 6-7, we will focus on:
- Introduction to Unity for robot visualization
- High-fidelity rendering and human-robot interaction
- Physics simulation and sensor simulation
- Creating interactive testing environments

## Resources
- [Unity Robotics Hub](https://unity.com/products/unity-robotics-hub)
- [ROS-TCP-Connector Documentation](https://github.com/Unity-Technologies/ROS-TCP-Connector)
- [URDF Importer Package](https://github.com/Unity-Technologies/URDF-Importer)
- [Unity XR Documentation](https://docs.unity3d.com/Manual/XR.html)
- [Unity Performance Guidelines](https://docs.unity3d.com/Manual/BestPracticeUnderstandingPerformanceInUnity.html)
