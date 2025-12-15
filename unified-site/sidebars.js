// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  bookSidebar: [
    'intro',
    'chapter1',
    {
      type: 'category',
      label: 'Module 1: ROS2',
      items: [
        'module-1-ros2/chapter-1-architecture',
        'module-1-ros2/chapter-2-rclpy',
        'module-1-ros2/chapter-3-packages',
        'module-1-ros2/chapter-4-urdf',
        'module-1-ros2/chapter-5-control-loops',
        'module-1-ros2/chapter-6-jetson-deployment',
      ],
    },
    {
      type: 'category',
      label: 'Module 2: Simulation',
      items: [
        'module-2-simulation/chapter-1-gazebo-setup',
        'module-2-simulation/chapter-2-urdf-gazebo-pipeline',
        'module-2-simulation/chapter-3-physics',
        'module-2-simulation/chapter-4-sensor-simulation',
        'module-2-simulation/chapter-5-unity-visuals',
        'module-2-simulation/chapter-6-interactive-testing',
      ],
    },
    {
      type: 'category',
      label: 'Module 3: AI Brain',
      items: [
        'module-3-ai-brain/chapter-1-isaac-sim-setup',
        'module-3-ai-brain/chapter-2-isaac-ros-pipelines',
        'module-3-ai-brain/chapter-3-nav2-bipedal-planning',
        'module-3-ai-brain/chapter-4-reinforcement-learning',
        'module-3-ai-brain/chapter-5-sim-to-real-concepts',
        'module-3-ai-brain/chapter-6-integrating-isaac-outputs',
      ],
    },
    {
      type: 'category',
      label: 'Module 4: VLA Systems',
      items: [
        'module-4-vla/chapter-1-vla-overview',
        'module-4-vla/chapter-2-voice-to-action',
        'module-4-vla/chapter-3-cognitive-planning',
        'module-4-vla/chapter-4-advanced-vla-applications',
        'module-4-vla/chapter-5-vision-language-grounding',
        'module-4-vla/chapter-6-safety-fallback-behaviors',
        'module-4-vla/chapter-7-capstone-autonomous-humanoid',
      ],
    },
    {
      type: 'category',
      label: 'Capstone Project',
      items: [
        'capstone-project/intro',
        'capstone-project/phase-1-robot-design',
        'capstone-project/phase-2-locomotion',
        'capstone-project/phase-3-perception',
        'capstone-project/implementation-guide',
      ],
    },
  ],
};

export default sidebars;