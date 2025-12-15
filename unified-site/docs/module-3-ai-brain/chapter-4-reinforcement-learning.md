# Chapter 4: Reinforcement Learning for Humanoid Control

## Introduction to Reinforcement Learning in Humanoid Robotics

Reinforcement Learning (RL) has emerged as a powerful paradigm for developing adaptive and robust control policies for humanoid robots. Unlike traditional control methods that rely on predefined mathematical models, RL enables robots to learn optimal behaviors through interaction with their environment. This approach is particularly valuable for humanoid robots, which operate in complex, dynamic environments and must handle the challenges of bipedal locomotion.

## Fundamentals of Reinforcement Learning

### The RL Framework
Reinforcement Learning operates within a Markov Decision Process (MDP) framework, characterized by:
- **State Space (S)**: The set of all possible states the robot can be in
- **Action Space (A)**: The set of possible actions the robot can take
- **Reward Function (R)**: A function that provides feedback on the quality of actions
- **Transition Dynamics (P)**: The probability distribution over next states given current state and action
- **Discount Factor (γ)**: A factor that determines the importance of future rewards

### Key RL Concepts
- **Policy (π)**: A mapping from states to actions that defines the agent's behavior
- **Value Function (V)**: The expected cumulative reward from a given state
- **Q-Function (Q)**: The expected cumulative reward from a state-action pair

## Challenges in Humanoid RL

### High-Dimensional Action Spaces
Humanoid robots typically have 20+ degrees of freedom, resulting in high-dimensional continuous action spaces that make traditional RL methods computationally intractable.

### Safety Constraints
Humanoid robots must maintain balance and avoid self-collision during learning, which requires careful reward shaping and exploration strategies.

### Real-Time Constraints
Humanoid locomotion requires real-time control at high frequencies (typically 100-500 Hz), making it challenging to apply RL algorithms that require extensive computation.

### Physical Safety
Damage to expensive hardware during exploration must be prevented, necessitating safe exploration strategies and simulation-to-real transfer methods.

## RL Algorithms for Humanoid Control

### Deep Deterministic Policy Gradient (DDPG)
DDPG is particularly well-suited for humanoid control due to its ability to handle continuous action spaces:

```python
import torch
import torch.nn as nn
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q = torch.relu(self.l1(sa))
        q = torch.relu(self.l2(q))
        return self.l3(q)
```

### Twin Delayed DDPG (TD3)
TD3 addresses overestimation bias in Q-learning, which is crucial for stable humanoid control:

```python
class TD3:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic_1 = Critic(state_dim, action_dim)
        self.critic_1_target = Critic(state_dim, action_dim)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters())

        self.critic_2 = Critic(state_dim, action_dim)
        self.critic_2_target = Critic(state_dim, action_dim)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters())

        self.max_action = max_action
        self.noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=100):
        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.noise).clamp(-self.noise_clip, self.noise_clip)

            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1 = self.critic_1(state, action)
        current_Q2 = self.critic_2(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize Critic
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic_1(state, self.actor(state)).mean()

            # Optimize Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target networks
            for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
```

### Soft Actor-Critic (SAC)
SAC provides maximum entropy RL, which is beneficial for exploration in complex humanoid tasks:

```python
class SAC:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = GaussianPolicy(state_dim, action_dim, max_action)
        self.critic = QNetwork(state_dim, action_dim)
        self.critic_target = QNetwork(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.alpha = 0.2  # Temperature parameter
        self.target_entropy = -torch.prod(torch.Tensor(action_dim)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)

    def train(self, replay_buffer, batch_size=256):
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_q_min = self.critic_target(next_state, next_action)
            target_q = reward + not_done * self.gamma * (target_q_min - self.alpha * next_log_prob)

        # Get current Q estimates
        current_q = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_q, target_q)

        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        pi, log_pi = self.actor.sample(state)
        q = self.critic(state, pi)
        actor_loss = (self.alpha * log_pi - q).mean()

        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update alpha
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()
```

## Humanoid-Specific RL Considerations

### State Representation
For humanoid robots, the state representation must include:
- Joint positions and velocities
- Center of Mass (CoM) position and velocity
- Inertial Measurement Unit (IMU) readings (acceleration, angular velocity)
- Contact information (ground contact sensors)
- Target position/direction for navigation tasks

```python
def get_humanoid_state(robot):
    """
    Construct the state vector for humanoid RL
    """
    state = []

    # Joint positions (relative to neutral pose)
    joint_positions = robot.get_joint_positions()
    state.extend(joint_positions)

    # Joint velocities
    joint_velocities = robot.get_joint_velocities()
    state.extend(joint_velocities)

    # IMU data
    imu_data = robot.get_imu_data()
    state.extend([imu_data['roll'], imu_data['pitch'], imu_data['yaw']])
    state.extend([imu_data['angular_vel_x'], imu_data['angular_vel_y'], imu_data['angular_vel_z']])
    state.extend([imu_data['linear_acc_x'], imu_data['linear_acc_y'], imu_data['linear_acc_z']])

    # Center of Mass information
    com_pos = robot.get_com_position()
    com_vel = robot.get_com_velocity()
    state.extend(com_pos)
    state.extend(com_vel)

    # Ground contact information
    contact_info = robot.get_contact_sensors()
    state.extend(contact_info)

    # Target information (for navigation tasks)
    target_pos = robot.get_target_position()
    relative_target = [target_pos[i] - com_pos[i] for i in range(2)]  # x, y only
    state.extend(relative_target)

    return np.array(state)
```

### Action Space Design
The action space for humanoid robots typically consists of:
- Desired joint positions (position control)
- Desired joint torques (torque control)
- Desired joint velocities (velocity control)

### Reward Engineering
Designing appropriate reward functions is crucial for successful humanoid RL:

```python
def humanoid_reward(robot, action, dt):
    """
    Compute reward for humanoid robot based on various factors
    """
    reward = 0.0

    # Forward progress reward
    forward_vel = robot.get_forward_velocity()
    reward += forward_vel * dt  # Encourage forward movement

    # Balance reward
    roll, pitch = robot.get_imu_orientation()
    balance_penalty = abs(roll) + abs(pitch)
    reward -= balance_penalty * 0.1  # Penalize deviation from upright position

    # Energy efficiency reward
    joint_velocities = robot.get_joint_velocities()
    energy_penalty = np.sum(np.abs(action))  # Penalize high control effort
    reward -= energy_penalty * 0.01

    # Smoothness reward
    prev_action = robot.get_previous_action()
    action_smoothness = np.sum(np.abs(action - prev_action))
    reward -= action_smoothness * 0.001

    # Survival reward
    if robot.is_in_safe_state():
        reward += 0.1  # Small reward for staying upright and stable

    # Task-specific rewards (e.g., reaching target)
    if hasattr(robot, 'target_reached') and robot.target_reached():
        reward += 10.0  # Large reward for completing task

    return reward
```

## Simulation Environments for Humanoid RL

### NVIDIA Isaac Gym
Isaac Gym provides GPU-accelerated physics simulation, enabling thousands of parallel environments for efficient RL training:

```python
import isaacgym
from isaacgym import gymapi
from isaacgym.torch_utils import *

class IsaacGymHumanoidEnv:
    def __init__(self, cfg):
        self.gym = gymapi.acquire_gym()

        # Configure sim
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

        # Create ground plane
        plane_params = gymapi.PlaneParams()
        self.gym.add_ground(self.sim, plane_params)

        # Create asset
        asset_root = cfg['asset']['root']
        asset_file = cfg['asset']['file']

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = False
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01

        self.humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # Set up camera
        self.camera_props = gymapi.CameraProperties()
        self.camera_props.width = 640
        self.camera_props.height = 480
        self.camera_handle = self.gym.create_camera_sensor(self.envs[0], self.camera_handles[0], self.camera_props)
```

### MuJoCo Integration
MuJoCo provides accurate physics simulation for humanoid control:

```python
import mujoco_py
from gym import Env, spaces

class HumanoidMujocoEnv(Env):
    def __init__(self):
        self.model = mujoco_py.load_model_from_path("humanoid_model.xml")
        self.sim = mujoco_py.MjSim(self.model)
        self.viewer = None

        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.get_obs().shape[0],), dtype=np.float32
        )

    def step(self, action):
        # Apply action to simulation
        self.sim.data.ctrl[:] = action
        self.sim.step()

        # Get observation
        obs = self.get_obs()

        # Calculate reward
        reward = self.compute_reward()

        # Check if episode is done
        done = self.is_done()

        return obs, reward, done, {}

    def reset(self):
        self.sim.reset()
        # Reset to initial state
        self.sim.data.qpos[:] = self.initial_qpos
        self.sim.data.qvel[:] = self.initial_qvel
        return self.get_obs()

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
        self.viewer.render()
```

## Transfer Learning: From Simulation to Reality

### Domain Randomization
Domain randomization helps policies generalize from simulation to reality by randomizing simulation parameters:

```python
def randomize_simulation_parameters(env):
    """
    Randomize physics parameters to improve sim-to-real transfer
    """
    # Randomize physical properties
    friction_range = [0.5, 1.5]
    mass_range = [0.8, 1.2]
    damping_range = [0.8, 1.2]

    # Apply randomization to each link
    for link_idx in range(env.num_bodies):
        # Randomize friction
        random_friction = np.random.uniform(friction_range[0], friction_range[1])
        env.set_friction(link_idx, random_friction)

        # Randomize mass
        original_mass = env.get_original_mass(link_idx)
        random_mass = original_mass * np.random.uniform(mass_range[0], mass_range[1])
        env.set_mass(link_idx, random_mass)

        # Randomize damping
        random_damping = np.random.uniform(damping_range[0], damping_range[1])
        env.set_damping(link_idx, random_damping)

    # Randomize actuator parameters
    for joint_idx in range(env.num_joints):
        random_gear_ratio = np.random.uniform(0.9, 1.1)
        env.set_actuator_gear_ratio(joint_idx, random_gear_ratio)
```

### System Identification
System identification techniques can help match simulation parameters to real robot dynamics:

```python
def system_identification(robot, sim_env):
    """
    Identify system parameters by comparing robot and simulation behavior
    """
    # Collect data from real robot
    real_data = collect_robot_data(robot)

    # Define objective function to minimize
    def objective(params):
        # Set simulation parameters
        sim_env.set_params(params)

        # Run simulation with same inputs as real robot
        sim_data = run_simulation(sim_env, real_data['inputs'])

        # Compute error between real and simulated behavior
        error = compute_error(real_data['outputs'], sim_data['outputs'])
        return error

    # Optimize parameters
    initial_params = sim_env.get_params()
    optimized_params = scipy.optimize.minimize(objective, initial_params)

    # Update simulation with optimized parameters
    sim_env.set_params(optimized_params)

    return optimized_params
```

## Advanced Techniques

### Hierarchical RL
Hierarchical RL decomposes complex humanoid tasks into simpler sub-tasks:

```python
class HierarchicalRL:
    def __init__(self, state_dim, action_dim, subgoal_dim):
        # High-level policy for subgoal selection
        self.high_level_policy = Actor(state_dim, subgoal_dim)

        # Low-level policy for primitive actions
        self.low_level_policy = Actor(state_dim + subgoal_dim, action_dim)

        # Subgoal generator
        self.subgoal_generator = SubgoalGenerator(state_dim, subgoal_dim)

    def select_action(self, state, time_step, subgoal_horizon):
        # Update subgoal periodically
        if time_step % subgoal_horizon == 0:
            self.current_subgoal = self.high_level_policy(state)

        # Generate action based on current state and subgoal
        extended_state = np.concatenate([state, self.current_subgoal])
        action = self.low_level_policy(extended_state)

        return action
```

### Multi-Task Learning
Training a single policy to handle multiple tasks:

```python
class MultiTaskHumanoidPolicy:
    def __init__(self, state_dim, action_dim, num_tasks):
        self.task_embedding = nn.Embedding(num_tasks, 64)  # Task-specific embedding
        self.policy_network = PolicyNetwork(state_dim + 64, action_dim)  # State + task embedding

    def forward(self, state, task_id):
        task_emb = self.task_embedding(task_id)
        extended_state = torch.cat([state, task_emb], dim=-1)
        action = self.policy_network(extended_state)
        return action
```

## Safety and Robustness

### Safe RL with Control Barrier Functions
Incorporate safety constraints using Control Barrier Functions (CBFs):

```python
def safe_action_selection(robot, unsafe_action, safety_constraints):
    """
    Modify unsafe action to satisfy safety constraints using QP
    """
    # Define safety constraints (e.g., joint limits, balance constraints)
    A, b = get_safety_constraints(robot, unsafe_action)

    # Solve QP to find safe action closest to desired action
    P = np.eye(len(unsafe_action))  # Minimize deviation from desired action
    q = -unsafe_action  # Minimize ||action - unsafe_action||^2

    # Solve: min 0.5 * x^T * P * x + q^T * x
    # subject to A * x <= b
    safe_action = solve_qp(P, q, A, b, unsafe_action)

    return safe_action
```

### Adversarial Training
Train policies to be robust against disturbances:

```python
def adversarial_training(robot_env, policy, adversary):
    """
    Train policy against adversarial disturbances
    """
    for episode in range(num_episodes):
        state = robot_env.reset()

        for step in range(max_steps):
            # Get action from policy
            action = policy.select_action(state)

            # Get adversarial disturbance
            disturbance = adversary.select_disturbance(state, action)

            # Apply both action and disturbance to environment
            next_state, reward, done, info = robot_env.step_with_disturbance(action, disturbance)

            # Train policy to handle disturbance
            policy.train_on_batch(state, action, reward, next_state, done)

            # Train adversary to create more challenging disturbances
            adversary.train(state, action, next_state, reward)

            state = next_state

            if done:
                break
```

## Evaluation Metrics

### Locomotion Performance
- **Speed**: Average forward velocity achieved
- **Stability**: Time spent in stable state vs. falling
- **Energy Efficiency**: Cost of transport (energy per unit weight per unit distance)
- **Robustness**: Ability to recover from disturbances

### Learning Performance
- **Sample Efficiency**: How quickly the policy improves with experience
- **Asymptotic Performance**: Final performance level achieved
- **Generalization**: Performance on unseen environments or conditions

## Troubleshooting Common Issues

### Non-Converging Training
- **Cause**: Poor reward design or hyperparameters
- **Solution**: Redesign reward function, tune learning rates, increase exploration

### Unstable Behavior
- **Cause**: High variance in policy updates
- **Solution**: Use value function regularization, reduce learning rate, add noise to actions

### Sim-to-Real Gap
- **Cause**: Mismatch between simulation and reality
- **Solution**: Domain randomization, system identification, robust training

## Summary

Reinforcement Learning offers a promising approach to developing adaptive and robust control policies for humanoid robots. By leveraging advanced RL algorithms like TD3 and SAC, and incorporating humanoid-specific considerations such as state representation and safety constraints, we can train policies that enable complex behaviors. The key to success lies in proper reward engineering, simulation-to-reality transfer techniques, and ensuring safety during both training and deployment. As the field continues to evolve, we can expect even more sophisticated approaches that will enable humanoid robots to learn and adapt in increasingly complex real-world environments.
