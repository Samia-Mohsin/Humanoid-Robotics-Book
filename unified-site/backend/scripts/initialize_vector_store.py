import asyncio
import os
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
import qdrant_client
from core.config import settings

async def initialize_vector_store():
    """
    Initialize the Qdrant vector store with book content
    This script should be run once to populate the vector database
    """
    print("Initializing vector store with book content...")

    # Initialize Qdrant client
    client = qdrant_client.QdrantClient(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        api_key=settings.QDRANT_API_KEY
    )

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    try:
        # Create or recreate the collection
        client.recreate_collection(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            vectors_config=qdrant_client.http.models.VectorParams(
                size=384,  # Size of the embeddings (all-MiniLM-L6-v2 produces 384-dim vectors)
                distance=qdrant_client.http.models.Distance.COSINE
            )
        )
        print(f"Collection '{settings.QDRANT_COLLECTION_NAME}' created/recreated")

        # Load documents from book content
        # In a real implementation, this would load from your actual book content
        # For now, we'll create some sample content

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
            # For this example, we'll treat each content block as a document
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

        print(f"Processing {len(split_documents)} document chunks...")

        # Create vector store with documents
        vector_store = Qdrant.from_documents(
            split_documents,
            embeddings,
            location=":memory:",  # Use in-memory for testing, or specify actual Qdrant server
            collection_name=settings.QDRANT_COLLECTION_NAME,
            client=client
        )

        print(f"Successfully loaded {len(split_documents)} document chunks into vector store")
        print("Vector store initialization complete!")

    except Exception as e:
        print(f"Error initializing vector store: {str(e)}")
        raise e

if __name__ == "__main__":
    asyncio.run(initialize_vector_store())