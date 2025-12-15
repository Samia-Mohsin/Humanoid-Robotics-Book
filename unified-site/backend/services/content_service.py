from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from models.user import User, UserPreferences
from core.config import settings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
import qdrant_client
import asyncio
import logging

logger = logging.getLogger(__name__)

class ContentService:
    def __init__(self):
        # Initialize Qdrant client for RAG
        try:
            self.qdrant_client = qdrant_client.QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                api_key=settings.QDRANT_API_KEY
            )

            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {str(e)}")
            self.qdrant_client = None
            self.embeddings = None

    async def load_book_content(self):
        """
        Load book content into the vector store
        This method is called during application startup
        """
        try:
            if not self.qdrant_client:
                logger.warning("Qdrant client not available, skipping content loading")
                return

            logger.info("Loading book content into vector store...")

            # Create or recreate the collection
            self.qdrant_client.recreate_collection(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                vectors_config=qdrant_client.http.models.VectorParams(
                    size=384,  # Size of the embeddings (all-MiniLM-L6-v2 produces 384-dim vectors)
                    distance=qdrant_client.http.models.Distance.COSINE
                )
            )
            logger.info(f"Collection '{settings.QDRANT_COLLECTION_NAME}' created/recreated")

            # Sample book content - in real implementation, load from actual book files
            sample_content = [
                {
                    "content": """# Introduction to Physical AI & Humanoid Robotics

Physical AI is an interdisciplinary field that combines artificial intelligence with physical systems, focusing on how AI agents can interact with and learn from the physical world. This field encompasses robotics, machine learning, computer vision, and control theory.

Humanoid robotics specifically deals with robots that have human-like characteristics and abilities. These robots are designed to interact with human environments and perform tasks in ways similar to humans.

Key concepts in Physical AI include:
- Embodied Cognition: The idea that intelligence emerges from the interaction between an agent and its physical environment
- Sensorimotor Learning: Learning through sensory input and motor output
- Multi-modal Perception: Processing information from multiple sensory modalities
- Adaptive Control: Control systems that can adapt to changing conditions""",
                    "source": "introduction.md"
                },
                {
                    "content": """# ROS 2 Fundamentals

ROS 2 (Robot Operating System 2) is a flexible framework for writing robot software. It's a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms.

## Key Concepts:
- Nodes: Processes that perform computation
- Topics: Named buses over which nodes exchange messages
- Services: Synchronous request/response communication
- Actions: Asynchronous goal-oriented communication
- Parameters: Configuration values that can be changed at runtime

## Creating a Node:
A node is an executable that uses ROS 2 to communicate with other nodes. Nodes are organized into packages, which provide structure and allow for reusable components.

## Communication Patterns:
ROS 2 supports several communication patterns including publish/subscribe, service calls, and action servers. Each pattern serves different use cases and provides different guarantees about message delivery and synchronization.""",
                    "source": "ros2_fundamentals.md"
                },
                {
                    "content": """# Simulation with Gazebo & Unity

Simulation is a crucial part of robotics development as it allows for testing and validation without the risk of damaging expensive hardware. Gazebo and Unity are two popular simulation environments used in robotics.

## Gazebo Simulation:
Gazebo provides realistic physics simulation, high-quality graphics, and convenient programmatic interfaces. It's widely used in the robotics community for testing algorithms before deploying to real robots.

Key features of Gazebo:
- Physics engine for realistic simulation
- Sensor simulation (cameras, LIDAR, IMU, etc.)
- Plugin system for custom functionality
- Integration with ROS/ROS 2

## Unity Simulation:
Unity offers advanced graphics capabilities and a user-friendly interface. It's particularly useful for creating complex visual environments and for human-robot interaction studies.

## Best Practices:
- Start with simple models and gradually increase complexity
- Validate simulation results with real-world tests
- Use simulation for rapid prototyping and testing
- Implement proper sensor noise models for realism""",
                    "source": "simulation.md"
                },
                {
                    "content": """# AI Brain with NVIDIA Isaac

The NVIDIA Isaac platform provides a comprehensive solution for developing AI-powered robots. It includes hardware, software, and simulation tools designed to accelerate robotics development.

## Isaac ROS:
Isaac ROS is a collection of hardware-accelerated software packages that help developers create perception and navigation applications. These packages leverage NVIDIA's GPU computing capabilities for real-time performance.

Key components:
- Perception: Object detection, segmentation, depth estimation
- Navigation: Path planning, obstacle avoidance
- Manipulation: Grasping, pick-and-place operations
- Simulation: Isaac Sim for realistic testing

## AI Models:
Isaac supports various AI models for robotics applications:
- Vision-based perception models
- Reinforcement learning agents
- Control systems with neural networks
- Multi-modal AI systems

## Development Workflow:
1. Design robot in Isaac Sim
2. Train AI models in simulation
3. Transfer to real robot
4. Fine-tune with real-world data""",
                    "source": "ai_brain.md"
                },
                {
                    "content": """# Vision-Language-Action Systems

Vision-Language-Action (VLA) systems represent the cutting edge of embodied AI, where robots can understand natural language commands, perceive their environment, and execute complex actions.

## Key Components:
- Vision Processing: Understanding visual input from cameras and sensors
- Language Understanding: Processing natural language commands and queries
- Action Execution: Converting high-level commands into low-level robot actions

## VLA Models:
Modern VLA models are large-scale neural networks that can:
- Follow natural language instructions
- Adapt to new tasks without explicit programming
- Learn from human demonstrations
- Generalize across different robot platforms

## Training Approaches:
- Supervised learning from human demonstrations
- Reinforcement learning in simulation
- Imitation learning from expert policies
- Multi-task learning across different domains

## Applications:
- Household robotics
- Industrial automation
- Healthcare assistance
- Educational robotics""",
                    "source": "vla_systems.md"
                }
            ]

            # Process each document
            documents = []
            for item in sample_content:
                from langchain.schema import Document
                doc = Document(
                    page_content=item["content"],
                    metadata={"source": item["source"]}
                )
                documents.append(doc)

            # Split documents into smaller chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            split_documents = text_splitter.split_documents(documents)

            logger.info(f"Processing {len(split_documents)} document chunks...")

            # Create vector store with documents
            vector_store = Qdrant.from_documents(
                split_documents,
                self.embeddings,
                location=":memory:",  # Use in-memory for testing, or specify actual Qdrant server
                collection_name=settings.QDRANT_COLLECTION_NAME,
                client=self.qdrant_client
            )

            logger.info(f"Successfully loaded {len(split_documents)} document chunks into vector store")
            logger.info("Book content loaded successfully!")

        except Exception as e:
            logger.error(f"Error loading book content: {str(e)}")
            raise e

    async def personalize_content(self, content: str, user_id: str, context: Dict[str, Any] = {}) -> str:
        """
        Personalize content based on user preferences and background
        """
        try:
            # In a real implementation, this would fetch user preferences from the database
            # and modify the content based on their experience level, programming languages, etc.

            # For now, we'll return a mock personalized content
            experience_level = context.get('experience_level', 'intermediate')

            if experience_level == 'beginner':
                return f"[BEGINNER-FRIENDLY VERSION]\n\n{content}\n\n*Explanation: This content has been adapted for beginners. Key concepts are explained in simpler terms with practical examples.*"
            elif experience_level == 'advanced':
                return f"[ADVANCED VERSION]\n\n{content}\n\n*Advanced notes: Additional technical depth and implementation details provided for experienced practitioners.*"
            else:
                return content

        except Exception as e:
            logger.error(f"Error personalizing content: {str(e)}")
            return content

    async def get_available_chapters(self) -> List[Dict[str, Any]]:
        """
        Get list of available chapters/modules in the book
        """
        try:
            # Mock response - in real implementation, this would come from database
            return [
                {
                    "id": "module1",
                    "title": "Module 1: ROS 2 Fundamentals",
                    "chapters": [
                        {"id": "ch1-1", "title": "Introduction to ROS 2"},
                        {"id": "ch1-2", "title": "Nodes and Topics"},
                        {"id": "ch1-3", "title": "Services and Actions"}
                    ]
                },
                {
                    "id": "module2",
                    "title": "Module 2: Simulation with Gazebo & Unity",
                    "chapters": [
                        {"id": "ch2-1", "title": "Gazebo Simulation Environment"},
                        {"id": "ch2-2", "title": "Unity Integration"},
                        {"id": "ch2-3", "title": "Physics Simulation"}
                    ]
                },
                {
                    "id": "module3",
                    "title": "Module 3: AI Brain with NVIDIA Isaac",
                    "chapters": [
                        {"id": "ch3-1", "title": "NVIDIA Isaac Overview"},
                        {"id": "ch3-2", "title": "Perception Pipeline"},
                        {"id": "ch3-3", "title": "Decision Making"}
                    ]
                },
                {
                    "id": "module4",
                    "title": "Module 4: VLA Systems",
                    "chapters": [
                        {"id": "ch4-1", "title": "Vision-Language-Action Models"},
                        {"id": "ch4-2", "title": "Embodied AI"},
                        {"id": "ch4-3", "title": "Robot Control Systems"}
                    ]
                },
                {
                    "id": "capstone",
                    "title": "Capstone Project",
                    "chapters": [
                        {"id": "cap-1", "title": "Project Planning"},
                        {"id": "cap-2", "title": "Implementation"},
                        {"id": "cap-3", "title": "Testing and Deployment"}
                    ]
                }
            ]
        except Exception as e:
            logger.error(f"Error getting available chapters: {str(e)}")
            return []

    async def get_chapter_content(self, chapter_id: str, user_id: str) -> Dict[str, Any]:
        """
        Get content for a specific chapter, potentially personalized
        """
        try:
            # Mock response - in real implementation, this would come from database or file system
            sample_content = f"# Chapter {chapter_id}\n\nThis is the content for chapter {chapter_id} of the Physical AI & Humanoid Robotics book. This would typically contain detailed explanations, code examples, diagrams, and exercises related to the topic.\n\n## Key Concepts\n- Concept 1\n- Concept 2\n- Concept 3\n\n## Exercises\nTry implementing the concepts covered in this chapter."

            return {
                "id": chapter_id,
                "title": f"Chapter {chapter_id}",
                "content": sample_content,
                "personalized": False,
                "metadata": {
                    "estimated_reading_time": "15 minutes",
                    "difficulty": "intermediate",
                    "prerequisites": ["basic programming", "mathematics"]
                }
            }
        except Exception as e:
            logger.error(f"Error getting chapter content: {str(e)}")
            return {
                "id": chapter_id,
                "title": f"Chapter {chapter_id}",
                "content": f"Error loading content for chapter {chapter_id}: {str(e)}",
                "personalized": False,
                "metadata": {}
            }