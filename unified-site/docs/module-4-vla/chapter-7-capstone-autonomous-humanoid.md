# Chapter 7: Capstone Autonomous Humanoid

## Introduction to the Autonomous Humanoid Capstone

The Autonomous Humanoid capstone project represents the culmination of all the knowledge and skills learned throughout the Physical AI & Humanoid Robotics course. This project integrates all four modules (ROS 2, Simulation, AI Brain, and VLA) into a cohesive system that demonstrates the complete pipeline from perception to action in humanoid robotics.

The capstone challenge is to create a simulated humanoid robot that can:
- Receive and understand voice commands
- Plan and navigate complex paths
- Identify and manipulate objects using computer vision
- Operate autonomously in dynamic environments
- Demonstrate embodied intelligence through multimodal interaction

## Project Architecture Overview

The autonomous humanoid system integrates components from all four modules into a unified architecture:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Autonomous Humanoid System                       │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────────┐    ┌──────────────────────┐    │
│  │  Voice      │    │  Cognitive      │    │  Action & Control    │    │
│  │  Command    │───▶│  Planning       │───▶│  System             │    │
│  │  Reception  │    │  & Reasoning    │    │                     │    │
│  └─────────────┘    └─────────────────┘    └──────────────────────┘    │
│         │                      │                          │            │
│         ▼                      ▼                          ▼            │
│  ┌─────────────┐    ┌─────────────────┐    ┌──────────────────────┐    │
│  │  Speech     │    │  World Model    │    │  Navigation &       │    │
│  │  Recognition│    │  & Perception   │    │  Manipulation       │    │
│  │  (Whisper)  │    │  (Vision)       │    │  (ROS 2 Actions)    │    │
│  └─────────────┘    └─────────────────┘    └──────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Core System Components

The system consists of several interconnected modules that work together to enable autonomous operation:

```python
import asyncio
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import time
import logging

class RobotState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    PLANNING = "planning"
    EXECUTING = "executing"
    ERROR = "error"
    SAFETY_STOP = "safety_stop"

@dataclass
class Command:
    text: str
    confidence: float
    timestamp: float
    source: str  # "voice", "text", etc.

@dataclass
class TaskPlan:
    id: str
    steps: List[Dict[str, Any]]
    priority: int
    status: str
    dependencies: List[str]

class AutonomousHumanoidSystem:
    def __init__(self):
        self.state = RobotState.IDLE
        self.logger = logging.getLogger("autonomous_humanoid")

        # Initialize subsystems
        self.voice_system = VoiceCommandSystem()
        self.vision_system = VisionPerceptionSystem()
        self.cognitive_system = CognitivePlanningSystem()
        self.action_system = ActionExecutionSystem()
        self.safety_system = SafetyMonitoringSystem()

        # Task management
        self.task_queue = asyncio.Queue()
        self.active_tasks = {}
        self.world_model = WorldModel()

        # Event callbacks
        self.command_callbacks = []
        self.state_callbacks = []

        # System status
        self.system_status = {
            "initialized": False,
            "components_ready": {},
            "last_command_time": 0,
            "uptime": 0
        }

    async def initialize(self):
        """Initialize all subsystems"""
        self.logger.info("Initializing autonomous humanoid system...")

        # Initialize all subsystems concurrently
        init_tasks = [
            self.voice_system.initialize(),
            self.vision_system.initialize(),
            self.cognitive_system.initialize(),
            self.action_system.initialize(),
            self.safety_system.initialize()
        ]

        results = await asyncio.gather(*init_tasks, return_exceptions=True)

        # Check initialization results
        components_ready = {
            "voice": not isinstance(results[0], Exception),
            "vision": not isinstance(results[1], Exception),
            "cognitive": not isinstance(results[2], Exception),
            "action": not isinstance(results[3], Exception),
            "safety": not isinstance(results[4], Exception)
        }

        self.system_status["components_ready"] = components_ready
        self.system_status["initialized"] = all(components_ready.values())

        if self.system_status["initialized"]:
            self.logger.info("All subsystems initialized successfully")
            self.state = RobotState.IDLE
        else:
            self.logger.error("Failed to initialize all subsystems")
            self.state = RobotState.ERROR

        return self.system_status["initialized"]

    async def start_listening(self):
        """Start listening for voice commands"""
        if not self.system_status["initialized"]:
            self.logger.error("System not initialized")
            return False

        self.state = RobotState.LISTENING
        self.logger.info("Starting to listen for voice commands")

        # Start voice command listener
        await self.voice_system.start_listening()

        return True

    async def process_command(self, command: Command):
        """Process a received command through the full pipeline"""
        self.logger.info(f"Processing command: {command.text}")

        # Update state
        previous_state = self.state
        self.state = RobotState.PROCESSING

        try:
            # 1. Update world model with current perception
            current_perception = await self.vision_system.get_current_perception()
            self.world_model.update_perception(current_perception)

            # 2. Parse and understand the command
            parsed_command = await self.cognitive_system.parse_command(
                command.text, self.world_model.get_state()
            )

            # 3. Generate task plan
            task_plan = await self.cognitive_system.generate_plan(
                parsed_command, self.world_model.get_state()
            )

            # 4. Validate plan safety
            if not await self.safety_system.validate_plan(task_plan, self.world_model.get_state()):
                self.logger.error("Plan failed safety validation")
                self.state = previous_state
                return False

            # 5. Execute the plan
            execution_result = await self.execute_plan(task_plan)

            # 6. Update world model with results
            self.world_model.update_execution_result(task_plan, execution_result)

            self.logger.info(f"Command completed: {command.text}")
            self.state = RobotState.IDLE
            return True

        except Exception as e:
            self.logger.error(f"Error processing command: {str(e)}")
            self.state = RobotState.ERROR
            return False

    async def execute_plan(self, plan: TaskPlan):
        """Execute a task plan with safety monitoring"""
        self.state = RobotState.EXECUTING

        execution_results = []

        for i, step in enumerate(plan.steps):
            # Check safety before each step
            if not await self.safety_system.is_safe_to_proceed(step, self.world_model.get_state()):
                self.logger.warning(f"Unsafe to proceed with step {i}: {step['action']}")
                break

            # Execute the step
            result = await self.action_system.execute_action(step)
            execution_results.append(result)

            # Update world model with step result
            self.world_model.update_step_result(step, result)

            # Check if execution should continue
            if not result.get('success', False):
                self.logger.warning(f"Step {i} failed: {result.get('error', 'Unknown error')}")
                break

        return execution_results

    def add_command_callback(self, callback: Callable[[Command], None]):
        """Add callback for command processing"""
        self.command_callbacks.append(callback)

    def add_state_callback(self, callback: Callable[[RobotState], None]):
        """Add callback for state changes"""
        self.state_callbacks.append(callback)

    async def shutdown(self):
        """Gracefully shutdown the system"""
        self.logger.info("Shutting down autonomous humanoid system...")

        # Stop voice system
        await self.voice_system.stop_listening()

        # Cancel any active tasks
        for task_id in list(self.active_tasks.keys()):
            task = self.active_tasks[task_id]
            if not task.done():
                task.cancel()

        # Shutdown subsystems
        shutdown_tasks = [
            self.voice_system.shutdown(),
            self.vision_system.shutdown(),
            self.cognitive_system.shutdown(),
            self.action_system.shutdown(),
            self.safety_system.shutdown()
        ]

        await asyncio.gather(*shutdown_tasks, return_exceptions=True)

        self.logger.info("All subsystems shut down")
```

## Voice Command System

The voice command system handles speech recognition and command parsing:

```python
import openai
import asyncio
import queue
import pyaudio
import wave
import threading
from typing import Dict, Any

class VoiceCommandSystem:
    def __init__(self):
        self.is_listening = False
        self.audio_queue = queue.Queue()
        self.command_queue = asyncio.Queue()
        self.logger = logging.getLogger("voice_system")
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 1024
        self.record_seconds = 5
        self.audio = pyaudio.PyAudio()

        # OpenAI API configuration
        self.openai_client = None  # Will be configured with API key

    async def initialize(self):
        """Initialize the voice command system"""
        self.logger.info("Initializing voice command system...")

        # Verify audio device availability
        try:
            # Test audio input
            stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )
            stream.close()
            self.logger.info("Audio device available")
        except Exception as e:
            self.logger.error(f"Audio device not available: {str(e)}")
            return False

        return True

    async def start_listening(self):
        """Start listening for voice commands"""
        self.is_listening = True
        self.logger.info("Voice command system started listening")

        # Start audio recording thread
        recording_thread = threading.Thread(target=self._record_audio_continuously)
        recording_thread.daemon = True
        recording_thread.start()

        # Process audio in async loop
        asyncio.create_task(self._process_audio_loop())

    def _record_audio_continuously(self):
        """Continuously record audio in a separate thread"""
        stream = self.audio.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )

        frames = []
        silence_threshold = 500  # Adjust based on your environment
        silence_count = 0
        max_silence_count = int(self.rate / self.chunk * 2)  # 2 seconds of silence

        while self.is_listening:
            data = stream.read(self.chunk)
            frames.append(data)

            # Check for silence to determine end of command
            audio_data = [abs(int.from_bytes(data[i:i+2], byteorder='little', signed=True))
                         for i in range(0, len(data), 2)]
            avg_amplitude = sum(audio_data) / len(audio_data)

            if avg_amplitude < silence_threshold:
                silence_count += 1
                if silence_count > max_silence_count and len(frames) > 10:
                    # End of command detected
                    self._save_audio_chunk(frames)
                    frames = []
                    silence_count = 0
            else:
                silence_count = 0

        stream.close()

    def _save_audio_chunk(self, frames):
        """Save recorded audio chunk to queue for processing"""
        wf = wave.open("temp_audio.wav", 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_size(self.audio_format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(frames))
        wf.close()

        # Add to processing queue
        self.audio_queue.put("temp_audio.wav")

    async def _process_audio_loop(self):
        """Process audio chunks asynchronously"""
        while self.is_listening:
            try:
                if not self.audio_queue.empty():
                    audio_file = self.audio_queue.get_nowait()
                    command = await self._transcribe_audio(audio_file)

                    if command and command.strip():
                        # Add to command queue for processing
                        await self.command_queue.put(Command(
                            text=command,
                            confidence=0.8,  # Placeholder - would be from actual transcription
                            timestamp=time.time(),
                            source="voice"
                        ))
                else:
                    await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
            except Exception as e:
                self.logger.error(f"Error in audio processing loop: {str(e)}")
                await asyncio.sleep(0.1)

    async def _transcribe_audio(self, audio_file: str) -> Optional[str]:
        """Transcribe audio file using OpenAI Whisper API"""
        try:
            if self.openai_client is None:
                # For demo purposes, return a mock transcription
                # In real implementation, use:
                # transcript = self.openai_client.audio.transcriptions.create(
                #     model="whisper-1",
                #     file=open(audio_file, "rb")
                # )
                # return transcript.text
                return "Pick up the red cup from the table"  # Mock response for demo

            # Real implementation would use OpenAI API
            # transcript = await self.openai_client.audio.transcriptions.create(
            #     model="whisper-1",
            #     file=open(audio_file, "rb")
            # )
            # return transcript.text

        except Exception as e:
            self.logger.error(f"Error transcribing audio: {str(e)}")
            return None

    async def stop_listening(self):
        """Stop listening for voice commands"""
        self.is_listening = False
        self.logger.info("Voice command system stopped listening")

    async def shutdown(self):
        """Shutdown the voice command system"""
        await self.stop_listening()
        self.audio.terminate()
        self.logger.info("Voice command system shutdown complete")
```

## Vision Perception System

The vision system handles object detection, scene understanding, and environmental perception:

```python
import cv2
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from typing import Dict, List, Any

class VisionPerceptionSystem:
    def __init__(self):
        self.logger = logging.getLogger("vision_system")
        self.clip_model = None
        self.clip_processor = None
        self.object_detector = None
        self.is_initialized = False

    async def initialize(self):
        """Initialize the vision system"""
        self.logger.info("Initializing vision system...")

        try:
            # Initialize CLIP model for vision-language understanding
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

            # Initialize object detector (using a mock for now, would be YOLO, DETR, etc. in practice)
            # self.object_detector = initialize_object_detector()

            self.is_initialized = True
            self.logger.info("Vision system initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error initializing vision system: {str(e)}")
            return False

    async def get_current_perception(self) -> Dict[str, Any]:
        """Get current environmental perception"""
        if not self.is_initialized:
            self.logger.error("Vision system not initialized")
            return {}

        # In a real implementation, this would capture from camera
        # For demo purposes, we'll return mock perception data
        perception = {
            "timestamp": time.time(),
            "objects": [
                {"id": "obj1", "class": "cup", "position": {"x": 1.0, "y": 0.5, "z": 0.0}, "color": "red", "confidence": 0.95},
                {"id": "obj2", "class": "table", "position": {"x": 0.0, "y": 0.0, "z": 0.0}, "color": "brown", "confidence": 0.98},
                {"id": "obj3", "class": "chair", "position": {"x": 1.5, "y": -0.5, "z": 0.0}, "color": "black", "confidence": 0.92}
            ],
            "humans": [
                {"id": "human1", "position": {"x": 2.0, "y": 0.0, "z": 0.0}, "orientation": "facing_robot", "confidence": 0.99}
            ],
            "navigation_map": {
                "free_space": [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 0.0}, {"x": 2.0, "y": 0.0}],
                "obstacles": [{"x": 1.5, "y": -0.5, "radius": 0.3}],
                "safe_zones": [{"x": -1.0, "y": 0.0, "radius": 1.0}]
            },
            "lighting_conditions": "well_lit",
            "camera_pose": {"x": 0.0, "y": 0.0, "z": 1.5, "rotation": [0, 0, 0, 1]}
        }

        return perception

    async def detect_objects(self, image: np.ndarray, classes: List[str] = None) -> List[Dict[str, Any]]:
        """Detect objects in an image"""
        if not self.is_initialized:
            return []

        # In real implementation, run object detection on the image
        # For demo, return mock detections
        return [
            {"class": "cup", "bbox": [100, 100, 200, 200], "confidence": 0.95, "position_3d": [1.0, 0.5, 0.0]},
            {"class": "table", "bbox": [50, 300, 350, 400], "confidence": 0.98, "position_3d": [0.0, 0.0, 0.0]}
        ]

    async def recognize_scene(self, image: np.ndarray) -> Dict[str, Any]:
        """Recognize the scene and provide context"""
        if not self.is_initialized:
            return {}

        # In real implementation, analyze the scene
        # For demo, return mock scene recognition
        return {
            "scene_type": "kitchen",
            "furniture": ["table", "chairs"],
            "objects_of_interest": ["cup", "plate"],
            "spatial_layout": "open_space_with_furniture"
        }

    async def track_objects(self, objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Track objects across frames"""
        # Implement object tracking logic
        # For demo, return the same objects
        return objects

    async def shutdown(self):
        """Shutdown the vision system"""
        self.is_initialized = False
        self.logger.info("Vision system shutdown complete")
```

## Cognitive Planning System

The cognitive system handles command understanding, planning, and reasoning:

```python
import re
from typing import Dict, List, Any
from dataclasses import dataclass
import networkx as nx

@dataclass
class ActionStep:
    action_type: str
    parameters: Dict[str, Any]
    preconditions: List[str]
    effects: List[str]
    priority: int = 1

class CognitivePlanningSystem:
    def __init__(self):
        self.logger = logging.getLogger("cognitive_system")
        self.action_library = self._initialize_action_library()
        self.is_initialized = False

    def _initialize_action_library(self) -> Dict[str, Any]:
        """Initialize the library of available actions"""
        return {
            "navigation": {
                "move_to_location": {
                    "parameters": ["target_location", "speed"],
                    "preconditions": ["robot_is_operational", "path_is_clear"],
                    "effects": ["robot_at_location", "energy_consumed"]
                },
                "avoid_obstacle": {
                    "parameters": ["obstacle_location", "alternative_path"],
                    "preconditions": ["obstacle_detected"],
                    "effects": ["obstacle_avoided", "path_recalculated"]
                }
            },
            "manipulation": {
                "grasp_object": {
                    "parameters": ["object_id", "grasp_pose"],
                    "preconditions": ["object_reachable", "gripper_available"],
                    "effects": ["object_grasped", "gripper_occupied"]
                },
                "place_object": {
                    "parameters": ["target_location", "release_pose"],
                    "preconditions": ["object_grasped"],
                    "effects": ["object_placed", "gripper_free"]
                }
            },
            "interaction": {
                "speak": {
                    "parameters": ["text", "volume"],
                    "preconditions": ["speech_system_available"],
                    "effects": ["message_delivered"]
                }
            }
        }

    async def initialize(self):
        """Initialize the cognitive system"""
        self.logger.info("Initializing cognitive planning system...")

        # Load any required models or data
        # In a real implementation, this might load LLM models, etc.

        self.is_initialized = True
        self.logger.info("Cognitive planning system initialized successfully")
        return True

    async def parse_command(self, command_text: str, world_state: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a natural language command into structured representation"""
        if not self.is_initialized:
            return {}

        # Extract action and target from command
        command_lower = command_text.lower()

        # Simple pattern matching for demo
        patterns = [
            (r'pick up (.+?) from (.+)', ('grasp_object', 'location')),
            (r'pick (.+?) from (.+)', ('grasp_object', 'location')),
            (r'grasp (.+?) from (.+)', ('grasp_object', 'location')),
            (r'go to (.+)', ('navigate', 'location')),
            (r'move to (.+)', ('navigate', 'location')),
            (r'bring (.+?) to (.+)', ('grasp_and_transport', 'object_location')),
            (r'clean (.+)', ('clean_area', 'location')),
            (r'help', ('provide_assistance', 'request'))
        ]

        parsed_command = {
            "original_text": command_text,
            "action_type": None,
            "targets": [],
            "locations": [],
            "confidence": 0.8  # Placeholder confidence
        }

        for pattern, (action_type, target_type) in patterns:
            match = re.search(pattern, command_lower)
            if match:
                parsed_command["action_type"] = action_type
                groups = match.groups()

                if target_type == "location":
                    parsed_command["locations"] = list(groups)
                elif target_type == "object_location":
                    parsed_command["targets"] = [groups[0]]
                    parsed_command["locations"] = [groups[1]]
                elif target_type == "request":
                    parsed_command["targets"] = ["help_request"]

                break

        # If no pattern matched, use a default action
        if parsed_command["action_type"] is None:
            parsed_command["action_type"] = "unknown"
            parsed_command["confidence"] = 0.3

        return parsed_command

    async def generate_plan(self, parsed_command: Dict[str, Any], world_state: Dict[str, Any]) -> TaskPlan:
        """Generate a task plan based on parsed command and world state"""
        if not self.is_initialized:
            return TaskPlan(id="empty", steps=[], priority=1, status="failed", dependencies=[])

        plan_id = f"plan_{int(time.time())}"
        steps = []

        action_type = parsed_command.get("action_type", "unknown")

        if action_type == "grasp_object":
            # Find the target object in the world state
            target_obj = self._find_object_in_world(parsed_command, world_state)

            if target_obj:
                # Navigate to object
                nav_step = ActionStep(
                    action_type="move_to_location",
                    parameters={"target_location": target_obj["position"]},
                    preconditions=["robot_is_operational"],
                    effects=["robot_at_object_location"],
                    priority=1
                )
                steps.append({"action": nav_step, "description": f"Navigate to {target_obj['class']}"})

                # Grasp the object
                grasp_step = ActionStep(
                    action_type="grasp_object",
                    parameters={"object_id": target_obj["id"], "grasp_pose": self._calculate_grasp_pose(target_obj)},
                    preconditions=["robot_at_object_location", "object_reachable"],
                    effects=["object_grasped"],
                    priority=2
                )
                steps.append({"action": grasp_step, "description": f"Grasp the {target_obj['class']}"})

        elif action_type == "navigate":
            target_location = self._find_location_in_world(parsed_command, world_state)
            if target_location:
                nav_step = ActionStep(
                    action_type="move_to_location",
                    parameters={"target_location": target_location},
                    preconditions=["robot_is_operational"],
                    effects=["robot_at_location"],
                    priority=1
                )
                steps.append({"action": nav_step, "description": f"Navigate to {target_location}"})

        elif action_type == "grasp_and_transport":
            # Find target object and destination
            target_obj = self._find_object_in_world(parsed_command, world_state)
            destination = self._find_location_in_world(parsed_command, world_state)

            if target_obj and destination:
                # Navigate to object
                nav_to_obj = ActionStep(
                    action_type="move_to_location",
                    parameters={"target_location": target_obj["position"]},
                    preconditions=["robot_is_operational"],
                    effects=["robot_at_object_location"],
                    priority=1
                )
                steps.append({"action": nav_to_obj, "description": f"Navigate to {target_obj['class']}"})

                # Grasp the object
                grasp_step = ActionStep(
                    action_type="grasp_object",
                    parameters={"object_id": target_obj["id"], "grasp_pose": self._calculate_grasp_pose(target_obj)},
                    preconditions=["robot_at_object_location", "object_reachable"],
                    effects=["object_grasped"],
                    priority=2
                )
                steps.append({"action": grasp_step, "description": f"Grasp the {target_obj['class']}"})

                # Navigate to destination
                nav_to_dest = ActionStep(
                    action_type="move_to_location",
                    parameters={"target_location": destination},
                    preconditions=["object_grasped"],
                    effects=["robot_at_destination"],
                    priority=3
                )
                steps.append({"action": nav_to_dest, "description": f"Navigate to destination with object"})

                # Place the object
                place_step = ActionStep(
                    action_type="place_object",
                    parameters={"target_location": destination, "release_pose": destination},
                    preconditions=["robot_at_destination", "object_grasped"],
                    effects=["object_placed", "gripper_free"],
                    priority=4
                )
                steps.append({"action": place_step, "description": f"Place the {target_obj['class']} at destination"})

        return TaskPlan(
            id=plan_id,
            steps=steps,
            priority=1,
            status="generated",
            dependencies=[]
        )

    def _find_object_in_world(self, parsed_command: Dict[str, Any], world_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find an object in the world state based on the parsed command"""
        objects = world_state.get("objects", [])

        # Look for object mentioned in command
        command_targets = parsed_command.get("targets", [])

        for target in command_targets:
            for obj in objects:
                if target in obj.get("class", "").lower() or target in obj.get("id", "").lower():
                    return obj

        # If not found by name, return the first matching object
        for obj in objects:
            for target in command_targets:
                if target in obj.get("class", "").lower():
                    return obj

        # If still not found, return the first object as a fallback
        return objects[0] if objects else None

    def _find_location_in_world(self, parsed_command: Dict[str, Any], world_state: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Find a location in the world state based on the parsed command"""
        # This would implement location resolution logic
        # For demo, return a default location
        return {"x": 1.0, "y": 1.0, "z": 0.0}

    def _calculate_grasp_pose(self, obj: Dict[str, Any]) -> Dict[str, float]:
        """Calculate appropriate grasp pose for an object"""
        # Calculate grasp pose based on object properties
        obj_pos = obj["position"]
        return {
            "x": obj_pos["x"],
            "y": obj_pos["y"] + 0.1,  # Slightly above the object
            "z": obj_pos["z"] + 0.2,  # Above the object
            "orientation": [0, 0, 0, 1]  # Default orientation
        }

    async def shutdown(self):
        """Shutdown the cognitive system"""
        self.is_initialized = False
        self.logger.info("Cognitive planning system shutdown complete")
```

## Action Execution System

The action execution system handles the actual execution of planned actions:

```python
class ActionExecutionSystem:
    def __init__(self):
        self.logger = logging.getLogger("action_system")
        self.ros2_interface = None  # Interface to ROS 2 for actual robot control
        self.is_initialized = False

    async def initialize(self):
        """Initialize the action execution system"""
        self.logger.info("Initializing action execution system...")

        # Initialize ROS 2 interface
        # In a real implementation, this would connect to ROS 2
        # self.ros2_interface = ROS2Interface()

        self.is_initialized = True
        self.logger.info("Action execution system initialized successfully")
        return True

    async def execute_action(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single action step"""
        if not self.is_initialized:
            return {"success": False, "error": "System not initialized"}

        action = step.get("action")
        if not action:
            return {"success": False, "error": "No action provided"}

        action_type = action.action_type
        parameters = action.parameters

        self.logger.info(f"Executing action: {action_type} with parameters: {parameters}")

        try:
            if action_type == "move_to_location":
                result = await self._execute_navigation_action(parameters)
            elif action_type == "grasp_object":
                result = await self._execute_manipulation_action(parameters)
            elif action_type == "place_object":
                result = await self._execute_placement_action(parameters)
            elif action_type == "speak":
                result = await self._execute_speech_action(parameters)
            else:
                result = {"success": False, "error": f"Unknown action type: {action_type}"}

        except Exception as e:
            self.logger.error(f"Error executing action {action_type}: {str(e)}")
            result = {"success": False, "error": str(e)}

        return result

    async def _execute_navigation_action(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute navigation action"""
        target_location = parameters.get("target_location", {})

        # In a real implementation, this would send navigation commands via ROS 2
        # For demo, simulate navigation
        self.logger.info(f"Navigating to location: {target_location}")

        # Simulate navigation process
        await asyncio.sleep(2)  # Simulate time for navigation

        return {
            "success": True,
            "action": "move_to_location",
            "result": f"Successfully navigated to {target_location}",
            "execution_time": 2.0
        }

    async def _execute_manipulation_action(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute manipulation action"""
        object_id = parameters.get("object_id")
        grasp_pose = parameters.get("grasp_pose")

        self.logger.info(f"Attempting to grasp object: {object_id}")

        # Simulate manipulation process
        await asyncio.sleep(1.5)  # Simulate time for grasping

        # Simulate success/failure based on some criteria
        success = True  # In real implementation, this would check actual robot feedback

        return {
            "success": success,
            "action": "grasp_object",
            "result": f"Grasp attempt for {object_id}" + (" succeeded" if success else " failed"),
            "execution_time": 1.5
        }

    async def _execute_placement_action(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute placement action"""
        target_location = parameters.get("target_location")

        self.logger.info(f"Attempting to place object at: {target_location}")

        # Simulate placement process
        await asyncio.sleep(1.0)  # Simulate time for placement

        return {
            "success": True,
            "action": "place_object",
            "result": f"Successfully placed object at {target_location}",
            "execution_time": 1.0
        }

    async def _execute_speech_action(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute speech action"""
        text = parameters.get("text", "")

        self.logger.info(f"Speaking: {text}")

        # In real implementation, this would use text-to-speech
        # For demo, just log the speech

        return {
            "success": True,
            "action": "speak",
            "result": f"Spoke: {text}",
            "execution_time": 0.5
        }

    async def shutdown(self):
        """Shutdown the action execution system"""
        self.is_initialized = False
        self.logger.info("Action execution system shutdown complete")
```

## Safety Monitoring System

The safety system ensures all actions are executed safely:

```python
class SafetyMonitoringSystem:
    def __init__(self):
        self.logger = logging.getLogger("safety_system")
        self.safety_rules = self._initialize_safety_rules()
        self.is_initialized = False

    def _initialize_safety_rules(self) -> Dict[str, Any]:
        """Initialize safety rules and constraints"""
        return {
            "collision_avoidance": {
                "minimum_distance": 0.3,  # meters
                "check_during_navigation": True,
                "check_during_manipulation": True
            },
            "human_safety": {
                "personal_space": 0.5,  # meters
                "speed_limits": {
                    "near_humans": 0.3,  # m/s
                    "normal": 0.5
                }
            },
            "robot_safety": {
                "joint_limits": {"shoulder": 1.57, "elbow": 2.0},  # radians
                "power_limits": {"max_current": 10.0},  # amps
                "temperature_limits": {"max_temp": 70.0}  # Celsius
            },
            "operational_safety": {
                "minimum_battery": 0.1,  # 10%
                "max_operation_time": 3600.0  # 1 hour
            }
        }

    async def initialize(self):
        """Initialize the safety monitoring system"""
        self.logger.info("Initializing safety monitoring system...")

        # Initialize safety monitoring components
        self.is_initialized = True
        self.logger.info("Safety monitoring system initialized successfully")
        return True

    async def validate_plan(self, plan: TaskPlan, world_state: Dict[str, Any]) -> bool:
        """Validate that a task plan is safe to execute"""
        if not self.is_initialized:
            self.logger.error("Safety system not initialized")
            return False

        self.logger.info(f"Validating plan: {plan.id}")

        for step in plan.steps:
            action = step.get("action")
            if not action:
                continue

            # Check safety for each action in the plan
            if not await self._is_action_safe(action, world_state):
                self.logger.warning(f"Action in plan is not safe: {action.action_type}")
                return False

        return True

    async def is_safe_to_proceed(self, step: Dict[str, Any], world_state: Dict[str, Any]) -> bool:
        """Check if it's safe to proceed with the next action"""
        if not self.is_initialized:
            return False

        action = step.get("action")
        if not action:
            return False

        return await self._is_action_safe(action, world_state)

    async def _is_action_safe(self, action: ActionStep, world_state: Dict[str, Any]) -> bool:
        """Check if a specific action is safe to execute"""
        action_type = action.action_type
        parameters = action.parameters

        # Check collision avoidance for navigation actions
        if action_type == "move_to_location":
            target_location = parameters.get("target_location", {})
            if not await self._check_navigation_safety(target_location, world_state):
                return False

        # Check human safety for all actions
        if not await self._check_human_safety(action, world_state):
            return False

        # Check robot safety limits
        if not await self._check_robot_safety(action, world_state):
            return False

        return True

    async def _check_navigation_safety(self, target_location: Dict[str, float],
                                     world_state: Dict[str, Any]) -> bool:
        """Check if navigation to target location is safe"""
        # Check for obstacles in the path
        obstacles = world_state.get("navigation_map", {}).get("obstacles", [])

        for obstacle in obstacles:
            distance = self._calculate_distance(target_location, obstacle)
            if distance < self.safety_rules["collision_avoidance"]["minimum_distance"]:
                self.logger.warning(f"Navigation path blocked by obstacle at distance {distance}")
                return False

        # Check if path goes through safe zones
        safe_zones = world_state.get("navigation_map", {}).get("safe_zones", [])
        if safe_zones:
            # For demo, just check if target is in a safe zone
            in_safe_zone = False
            for zone in safe_zones:
                distance = self._calculate_distance(target_location, zone)
                if distance <= zone.get("radius", 0.5):
                    in_safe_zone = True
                    break

            if not in_safe_zone:
                self.logger.info("Target location not in predefined safe zone")

        return True

    async def _check_human_safety(self, action: ActionStep, world_state: Dict[str, Any]) -> bool:
        """Check if action is safe regarding humans in environment"""
        humans = world_state.get("humans", [])
        if not humans:
            return True

        # Check if action would violate personal space
        for human in humans:
            human_pos = human.get("position", {})

            # For navigation actions, check if path goes too close to humans
            if action.action_type == "move_to_location":
                target_pos = action.parameters.get("target_location", {})
                distance = self._calculate_distance(target_pos, human_pos)

                if distance < self.safety_rules["human_safety"]["personal_space"]:
                    self.logger.warning(f"Navigation would violate personal space: {distance}m")
                    return False

        return True

    async def _check_robot_safety(self, action: ActionStep, world_state: Dict[str, Any]) -> bool:
        """Check if action is safe for the robot itself"""
        # Check battery level
        battery_level = world_state.get("robot_state", {}).get("battery_level", 1.0)
        if battery_level < self.safety_rules["operational_safety"]["minimum_battery"]:
            self.logger.warning(f"Robot battery level too low: {battery_level}")
            return False

        # For manipulation actions, check if robot state is safe
        if action.action_type in ["grasp_object", "place_object"]:
            # Check if robot is in a safe configuration
            joint_angles = world_state.get("robot_state", {}).get("joint_angles", {})

            # Check joint limits (simplified check)
            for joint, angle in joint_angles.items():
                if abs(angle) > self.safety_rules["robot_safety"]["joint_limits"].get(joint, 3.14):
                    self.logger.warning(f"Joint limit violation for {joint}: {angle}")
                    return False

        return True

    def _calculate_distance(self, pos1: Dict[str, float], pos2: Dict[str, float]) -> float:
        """Calculate Euclidean distance between two 3D positions"""
        dx = pos1.get('x', 0) - pos2.get('x', 0)
        dy = pos1.get('y', 0) - pos2.get('y', 0)
        dz = pos1.get('z', 0) - pos2.get('z', 0)
        return (dx*dx + dy*dy + dz*dz)**0.5

    async def shutdown(self):
        """Shutdown the safety monitoring system"""
        self.is_initialized = False
        self.logger.info("Safety monitoring system shutdown complete")
```

## World Model

The world model maintains the robot's understanding of its environment:

```python
from datetime import datetime
from typing import Optional

class WorldModel:
    def __init__(self):
        self.logger = logging.getLogger("world_model")
        self.objects = {}
        self.humans = {}
        self.robot_state = {}
        self.navigation_map = {}
        self.last_update_time = datetime.now()
        self.confidence_threshold = 0.7

    def update_perception(self, perception_data: Dict[str, Any]):
        """Update world model with new perception data"""
        self.last_update_time = datetime.now()

        # Update objects
        new_objects = perception_data.get("objects", [])
        for obj in new_objects:
            obj_id = obj.get("id")
            if obj_id:
                # Update or add object
                if obj_id in self.objects:
                    # Merge with existing object data
                    self.objects[obj_id].update(obj)
                else:
                    self.objects[obj_id] = obj

        # Remove objects that are no longer detected (with some hysteresis)
        detected_ids = {obj.get("id") for obj in new_objects if obj.get("id")}
        for obj_id in list(self.objects.keys()):
            if obj_id not in detected_ids:
                # Keep object for a while before removing (to handle temporary occlusions)
                self.objects[obj_id]["last_seen"] = datetime.now()

        # Update humans
        humans = perception_data.get("humans", [])
        for human in humans:
            human_id = human.get("id")
            if human_id:
                self.humans[human_id] = human

        # Update navigation map
        nav_map = perception_data.get("navigation_map", {})
        self.navigation_map.update(nav_map)

        # Update robot state
        robot_state = perception_data.get("robot_state", {})
        self.robot_state.update(robot_state)

        self.logger.debug(f"World model updated with {len(new_objects)} objects and {len(humans)} humans")

    def update_step_result(self, step: Dict[str, Any], result: Dict[str, Any]):
        """Update world model with result of executed step"""
        # Update based on action effects
        action = step.get("action")
        if not action:
            return

        # For grasping actions, update object state
        if action.action_type == "grasp_object":
            obj_id = action.parameters.get("object_id")
            if obj_id and obj_id in self.objects:
                self.objects[obj_id]["grasped"] = True
                self.objects[obj_id]["location"] = "robot_gripper"

        # For placement actions
        elif action.action_type == "place_object":
            # Find object that was being held
            for obj_id, obj_data in self.objects.items():
                if obj_data.get("grasped"):
                    obj_data["grasped"] = False
                    obj_data["location"] = action.parameters.get("target_location")
                    break

    def update_execution_result(self, plan: TaskPlan, results: List[Dict[str, Any]]):
        """Update world model with full execution results"""
        self.logger.info(f"Updating world model with execution results for plan {plan.id}")

        # Process each step result
        for i, result in enumerate(results):
            if i < len(plan.steps):
                self.update_step_result(plan.steps[i], result)

    def get_object_by_class(self, obj_class: str) -> Optional[Dict[str, Any]]:
        """Get an object by its class"""
        for obj_id, obj_data in self.objects.items():
            if obj_data.get("class", "").lower() == obj_class.lower():
                # Only return objects with sufficient confidence
                if obj_data.get("confidence", 0) >= self.confidence_threshold:
                    return obj_data
        return None

    def get_objects_in_area(self, center: Dict[str, float], radius: float) -> List[Dict[str, Any]]:
        """Get objects within a certain area"""
        nearby_objects = []

        for obj_id, obj_data in self.objects.items():
            obj_pos = obj_data.get("position", {})
            distance = self._calculate_distance(center, obj_pos)

            if distance <= radius:
                nearby_objects.append(obj_data)

        return nearby_objects

    def get_state(self) -> Dict[str, Any]:
        """Get current world state"""
        return {
            "objects": list(self.objects.values()),
            "humans": list(self.humans.values()),
            "robot_state": self.robot_state,
            "navigation_map": self.navigation_map,
            "timestamp": self.last_update_time
        }

    def _calculate_distance(self, pos1: Dict[str, float], pos2: Dict[str, float]) -> float:
        """Calculate distance between two positions"""
        dx = pos1.get('x', 0) - pos2.get('x', 0)
        dy = pos1.get('y', 0) - pos2.get('y', 0)
        dz = pos1.get('z', 0) - pos2.get('z', 0)
        return (dx*dx + dy*dy + dz*dz)**0.5
```

## System Integration and Testing

The complete system integration for the capstone project:

```python
class CapstoneDemo:
    """Demonstration of the complete autonomous humanoid system"""

    def __init__(self):
        self.system = AutonomousHumanoidSystem()
        self.logger = logging.getLogger("capstone_demo")

    async def run_demo(self):
        """Run a complete demonstration of the system"""
        self.logger.info("Starting autonomous humanoid capstone demonstration...")

        # Initialize the system
        if not await self.system.initialize():
            self.logger.error("Failed to initialize the system")
            return False

        self.logger.info("System initialized successfully")

        # Define demo commands
        demo_commands = [
            "Pick up the red cup from the table",
            "Go to the kitchen counter",
            "Bring the cup to the living room"
        ]

        for i, command_text in enumerate(demo_commands):
            self.logger.info(f"Processing demo command {i+1}: {command_text}")

            # Create command object
            command = Command(
                text=command_text,
                confidence=0.9,
                timestamp=time.time(),
                source="demo"
            )

            # Process the command
            success = await self.system.process_command(command)

            if success:
                self.logger.info(f"Command {i+1} completed successfully")
            else:
                self.logger.error(f"Command {i+1} failed")
                break

            # Small delay between commands
            await asyncio.sleep(1)

        self.logger.info("Capstone demonstration completed")
        return True

    async def run_continuous_demo(self):
        """Run a continuous demonstration with simulated voice commands"""
        self.logger.info("Starting continuous autonomous humanoid demonstration...")

        # Initialize the system
        if not await self.system.initialize():
            self.logger.error("Failed to initialize the system")
            return False

        # Simulate continuous operation
        demo_commands = [
            ("Navigate to the kitchen", 10),
            ("Find and pick up a cup", 15),
            ("Move to the dining table", 12),
            ("Place the cup on the table", 8),
            ("Return to the charging station", 15)
        ]

        for command_text, expected_time in demo_commands:
            self.logger.info(f"Executing: {command_text}")

            command = Command(
                text=command_text,
                confidence=0.85,
                timestamp=time.time(),
                source="continuous_demo"
            )

            start_time = time.time()
            success = await self.system.process_command(command)
            execution_time = time.time() - start_time

            if success:
                self.logger.info(f"Completed in {execution_time:.2f}s (expected ~{expected_time}s)")
            else:
                self.logger.error(f"Failed to execute: {command_text}")
                break

            # Wait a bit before next command
            await asyncio.sleep(2)

        self.logger.info("Continuous demonstration completed")
        return True

async def main():
    """Main entry point for the capstone system"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("capstone_main")

    logger.info("Initializing Autonomous Humanoid Capstone System")

    demo = CapstoneDemo()

    # Run the demonstration
    success = await demo.run_demo()

    if success:
        logger.info("Demo completed successfully!")
    else:
        logger.error("Demo failed!")

    # Also run continuous demo
    logger.info("\nRunning continuous demonstration...")
    continuous_success = await demo.run_continuous_demo()

    if continuous_success:
        logger.info("Continuous demo completed successfully!")
    else:
        logger.error("Continuous demo failed!")

    logger.info("Capstone system demonstration finished")

# Example usage
if __name__ == "__main__":
    asyncio.run(main())
```

## Conclusion

The Autonomous Humanoid capstone project demonstrates the integration of all the core technologies learned throughout the Physical AI & Humanoid Robotics course. This system combines:

1. **ROS 2 Communication**: For reliable message passing between components
2. **Simulation & Perception**: For environment understanding and object detection
3. **AI Brain**: For cognitive planning and decision making
4. **Vision-Language-Action**: For natural human-robot interaction

The project showcases how these technologies work together to create an autonomous system capable of receiving voice commands, understanding its environment, planning appropriate actions, and executing them safely. The modular architecture allows for easy extension and modification, making it a solid foundation for further development.

Key achievements of this capstone project include:
- End-to-end integration of all course modules
- Safe operation with comprehensive safety monitoring
- Natural language interaction capabilities
- Robust perception and planning systems
- Scalable architecture for future enhancements

This capstone project serves as a demonstration of the skills and knowledge acquired throughout the course, showing how to build a complete, functional autonomous humanoid robot system.
