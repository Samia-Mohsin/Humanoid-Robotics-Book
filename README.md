# Physical AI & Humanoid Robotics Educational Platform

[![Vercel Deployment](https://vercelbadge.vercel.app/api/samia-mohsin/Humanoid-Robotics-Book)](https://humanoid-robotics-book-xi.vercel.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Docusaurus 3](https://img.shields.io/badge/Docusaurus-3.1.0-informational?logo=docusaurus&color=red)](https://docusaurus.io/)

## 🤖 Project Overview

An AI/Spec-Driven educational platform for **Physical AI & Humanoid Robotics** built with Docusaurus 3, featuring an integrated RAG (Retrieval-Augmented Generation) chatbot powered by OpenAI, Cohere embeddings, and Qdrant vector database. This project was created using Spec-Kit Plus and Claude Code following a spec-driven development approach.

### 🎯 Course Focus: Physical AI & Humanoid Robotics
- **Theme**: AI Systems in the Physical World. Embodied Intelligence.
- **Goal**: Bridging the gap between the digital brain and the physical body.
- **Students**: Apply AI knowledge to control Humanoid Robots in simulated and real-world environments.

## 📚 Complete Course Curriculum

### 🧠 Module 1: The Robotic Nervous System (ROS 2)
- ROS 2 Nodes, Topics, and Services
- Bridging Python Agents to ROS controllers using rclpy
- Understanding URDF (Unified Robot Description Format) for humanoids
- Middleware for robot control fundamentals

### 🌐 Module 2: The Digital Twin (Gazebo & Unity)
- Physics simulation and environment building
- Simulating physics, gravity, and collisions in Gazebo
- High-fidelity rendering and human-robot interaction in Unity
- Sensor simulation: LiDAR, Depth Cameras, and IMUs

### 🤖 Module 3: The AI-Robot Brain (NVIDIA Isaac™)
- NVIDIA Isaac Sim: Photorealistic simulation and synthetic data generation
- Isaac ROS: Hardware-accelerated VSLAM (Visual SLAM) and navigation
- Nav2: Path planning for bipedal humanoid movement
- Advanced perception and training systems

### 🗣️ Module 4: Vision-Language-Action (VLA)
- Voice-to-Action: Using OpenAI Whisper for voice commands
- Cognitive Planning: Using LLMs to translate natural language to ROS 2 actions
- Convergence of LLMs and Robotics
- Capstone: Autonomous Humanoid Project

### 🏆 Capstone Project: The Autonomous Humanoid
A final project where a simulated robot receives a voice command, plans a path, navigates obstacles, identifies an object using computer vision, and manipulates it.

## 🚀 Key Features

### 💬 Interactive RAG Chatbot
- **Floating Chat Bubble**: Always accessible in bottom-right corner
- **Text Selection Capture**: Automatically captures selected text as context
- **RAG-Powered Responses**: Answers based on book content with proper citations
- **Context-Aware**: Prioritizes selected text when answering questions

### 🎨 Chapter Header Features
- **"Personalize for Me"**: Adapts content based on user profile and preferences
- **"اردو میں ترجمہ"**: Translates content to Urdu with proper RTL support
- **Progress Circle**: Tracks completion status with visual indicators
- **Authentication Required**: Only visible for logged-in users

### 🔐 Authentication & Personalization
- **Better-Auth Integration**: Secure authentication system
- **User Profiles**: Background questions during signup (experience level, hardware, etc.)
- **Adaptive Content**: Personalizes learning experience based on user preferences
- **Progress Tracking**: Saves learning progress across sessions

### 🌍 Multilingual Support
- **English Content**: Primary educational content
- **Urdu Translation**: Full chapter translation capability
- **Roman Urdu Support**: Alternative transliteration option
- **RTL Layout**: Proper right-to-left rendering for Urdu content

## 🛠️ Technical Architecture

### Frontend Stack
- **Framework**: Docusaurus 3 (latest)
- **Language**: React 18 + TypeScript
- **Styling**: Tailwind CSS + shadcn/ui components
- **Authentication**: @better-auth/react
- **State Management**: React Context API
- **Internationalization**: Docusaurus i18n with Urdu support

### Backend Stack
- **Framework**: FastAPI with Uvicorn
- **Database**: PostgreSQL with SQLAlchemy 2.0 + asyncpg
- **Vector DB**: Qdrant Client for vector storage
- **AI Integration**: LangChain 0.3+, OpenAI SDK, Cohere
- **Authentication**: @better-auth/fastapi
- **API Documentation**: Automatic OpenAPI/Swagger generation

### RAG System
- **Embeddings**: Cohere multilingual models (v3.0)
- **Vector Storage**: Qdrant Cloud Free Tier
- **Retrieval**: Semantic search with configurable similarity thresholds
- **Generation**: OpenAI GPT models with context-aware prompting
- **Caching**: Redis-based caching for frequent queries

## 📁 Project Structure

```
unified-site/                    # Main Docusaurus application
├── docs/                       # Complete course content
│   ├── intro.md                # Introduction
│   ├── chapter1.md             # Getting Started
│   ├── module-1-ros2/          # Module 1: ROS 2 content
│   ├── module-2-simulation/    # Module 2: Simulation content
│   ├── module-3-ai-brain/      # Module 3: AI Brain content
│   ├── module-4-vla/           # Module 4: VLA Systems content
│   └── capstone-project/       # Capstone project content
├── src/                        # Custom React components
│   ├── components/             # Reusable UI components
│   │   ├── Chatbot/            # Floating chat interface
│   │   ├── Chapter/            # Chapter header components
│   │   └── Navbar.jsx          # Custom navigation
│   ├── pages/                  # Additional pages
│   └── contexts/               # React context providers
├── backend/                    # FastAPI backend
│   ├── main.py                 # Application entry point
│   ├── api/                    # API route definitions
│   ├── models/                 # Database models
│   ├── services/               # Business logic
│   ├── schemas/                # Pydantic schemas
│   └── core/                   # Core utilities
├── static/                     # Static assets
└── package.json               # Frontend dependencies
```

## 🚀 Quick Start

### Prerequisites
- Node.js >= 18.0
- Python >= 3.8
- PostgreSQL (for backend)
- OpenAI API Key
- Cohere API Key
- Qdrant Cloud Account

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/Samia-Mohsin/AI-Textbook.git
cd AI-Textbook
```

2. **Install frontend dependencies**:
```bash
cd unified-site
npm install
```

3. **Install backend dependencies**:
```bash
cd backend
pip install -r requirements.txt
```

4. **Set up environment variables**:
```bash
cp .env.example .env
# Edit .env with your API keys and database configuration
```

5. **Run the development servers**:
```bash
# Terminal 1: Start backend
cd unified-site/backend
python -m uvicorn main:app --reload

# Terminal 2: Start frontend
cd unified-site
npm start
```

### Production Build
```bash
cd unified-site
npm run build
```

## 📊 Learning Outcomes

Upon completion of this course, students will be able to:
1. Understand Physical AI principles and embodied intelligence
2. Master ROS 2 (Robot Operating System) for robotic control
3. Simulate robots with Gazebo and Unity
4. Develop with NVIDIA Isaac AI robot platform
5. Design humanoid robots for natural interactions
6. Integrate GPT models for conversational robotics

## 🏗️ Weekly Breakdown

| Weeks | Topic | Focus |
|-------|-------|-------|
| 1-2 | Introduction to Physical AI | Foundations of Physical AI and embodied intelligence |
| 3-5 | ROS 2 Fundamentals | Core concepts, nodes, topics, services |
| 6-7 | Robot Simulation | Gazebo, URDF, physics simulation |
| 8-10 | NVIDIA Isaac Platform | Perception, navigation, reinforcement learning |
| 11-12 | Humanoid Robot Development | Kinematics, locomotion, interaction |
| 13 | Conversational Robotics | GPT integration, speech recognition |

## 📋 Assessments

- **ROS 2 package development project**
- **Gazebo simulation implementation**
- **Isaac-based perception pipeline**
- **Capstone: Simulated humanoid robot with conversational AI**

## 🖥️ Hardware Requirements

### Digital Twin Workstation (Required per Student)
- **GPU**: NVIDIA RTX 4070 Ti (12GB VRAM) or higher
- **CPU**: Intel Core i7 (13th Gen+) or AMD Ryzen 9
- **RAM**: 64 GB DDR5 (32 GB minimum)
- **OS**: Ubuntu 22.04 LTS

### Physical AI Edge Kit
- **Brain**: NVIDIA Jetson Orin Nano (8GB) or Orin NX (16GB)
- **Eyes**: Intel RealSense D435i or D455
- **Inner Ear**: USB IMU (BNO055)
- **Voice Interface**: USB Microphone/Speaker array

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for more details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support, please open an issue in the GitHub repository or contact the development team.

---

Built with ❤️ using [Docusaurus](https://docusaurus.io/), [FastAPI](https://fastapi.tiangolo.com/), and [Claude Code](https://www.claude.com/product/claude-code)

🤖 **AI/Spec-Driven Development** powered by Spec-Kit Plus