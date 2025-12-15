import React, { useState } from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import FloatingChat from '@site/src/components/Chatbot/FloatingChat';
import ChapterActions from '@site/src/components/Chapter/ChapterActions';

function BookPage() {
  const { siteConfig } = useDocusaurusContext();
  const [content, setContent] = useState(`# Introduction to Physical AI & Humanoid Robotics

Physical AI is an interdisciplinary field that combines artificial intelligence with physical systems, focusing on how AI agents can interact with and learn from the physical world. This field encompasses robotics, machine learning, computer vision, and control theory.

Humanoid robotics specifically deals with robots that have human-like characteristics and abilities. These robots are designed to interact with human environments and perform tasks in ways similar to humans.

Key concepts in Physical AI include:
- Embodied Cognition: The idea that intelligence emerges from the interaction between an agent and its physical environment
- Sensorimotor Learning: Learning through sensory input and motor output
- Multi-modal Perception: Processing information from multiple sensory modalities
- Adaptive Control: Control systems that can adapt to changing conditions

## ROS 2 Fundamentals

ROS 2 (Robot Operating System 2) is a flexible framework for writing robot software. It's a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms.

### Key Concepts:
- Nodes: Processes that perform computation
- Topics: Named buses over which nodes exchange messages
- Services: Synchronous request/response communication
- Actions: Asynchronous goal-oriented communication
- Parameters: Configuration values that can be changed at runtime

### Creating a Node:
A node is an executable that uses ROS 2 to communicate with other nodes. Nodes are organized into packages, which provide structure and allow for reusable components.

### Communication Patterns:
ROS 2 supports several communication patterns including publish/subscribe, service calls, and action servers. Each pattern serves different use cases and provides different guarantees about message delivery and synchronization.

## Simulation with Gazebo & Unity

Simulation is a crucial part of robotics development as it allows for testing and validation without the risk of damaging expensive hardware. Gazebo and Unity are two popular simulation environments used in robotics.

### Gazebo Simulation:
Gazebo provides realistic physics simulation, high-quality graphics, and convenient programmatic interfaces. It's widely used in the robotics community for testing algorithms before deploying to real robots.

Key features of Gazebo:
- Physics engine for realistic simulation
- Sensor simulation (cameras, LIDAR, IMU, etc.)
- Plugin system for custom functionality
- Integration with ROS/ROS 2

### Unity Simulation:
Unity offers advanced graphics capabilities and a user-friendly interface. It's particularly useful for creating complex visual environments and for human-robot interaction studies.

### Best Practices:
- Start with simple models and gradually increase complexity
- Validate simulation results with real-world tests
- Use simulation for rapid prototyping and testing
- Implement proper sensor noise models for realism`);

  const handlePersonalize = (personalizedContent) => {
    setContent(personalizedContent);
  };

  const handleTranslate = (translatedContent) => {
    setContent(translatedContent);
  };

  return (
    <Layout
      title={`Physical AI & Humanoid Robotics`}
      description="AI-powered educational platform for Physical AI & Humanoid Robotics">
      <FloatingChat />
      <div className="container padding-vert--lg">
        <div className="row">
          <div className="col col--12">
            <ChapterActions
              chapterId="intro-physical-ai"
              chapterTitle="Introduction to Physical AI & Humanoid Robotics"
            />

            <div className="book-content">
              <div
                className="markdown"
                dangerouslySetInnerHTML={{ __html: content }}
              />
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
}

export default BookPage;