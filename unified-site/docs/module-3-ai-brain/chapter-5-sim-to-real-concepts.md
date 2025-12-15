# Chapter 5: Simulation-to-Reality Transfer Concepts

## Introduction to Sim-to-Real Transfer

Simulation-to-reality (sim-to-real) transfer is a critical aspect of developing humanoid robots, as it allows us to leverage the safety, speed, and cost-effectiveness of simulation for training and testing before deploying on physical hardware. The goal is to train policies in simulation that can successfully transfer to real robots with minimal additional training. However, the "reality gap" between simulation and reality presents significant challenges that must be addressed.

## Understanding the Reality Gap

### Definition and Causes
The reality gap refers to the differences between simulated and real environments that can cause policies trained in simulation to fail when deployed on real robots. These differences include:

- **Dynamics Mismatch**: Differences in friction, mass, damping, and other physical parameters
- **Sensor Noise**: Real sensors have noise, latency, and biases not present in simulation
- **Actuator Differences**: Real actuators have delays, limited bandwidth, and non-linear responses
- **Model Imperfections**: Simplified physics models in simulation vs. complex real-world physics
- **Environmental Factors**: Lighting, temperature, and other environmental conditions

### Quantifying the Reality Gap
```python
def quantify_reality_gap(sim_policy_performance, real_policy_performance):
    """
    Quantify the reality gap by comparing policy performance in simulation vs. reality
    """
    gap_metrics = {
        'performance_gap': sim_policy_performance - real_policy_performance,
        'variance_ratio': real_policy_variance / sim_policy_variance,
        'success_rate_drop': (sim_success_rate - real_success_rate) / sim_success_rate,
        'stability_difference': measure_stability_difference(sim_trajectory, real_trajectory)
    }
    return gap_metrics
```

## Domain Randomization

### Concept and Implementation
Domain randomization is a technique that randomizes simulation parameters during training to improve robustness to the reality gap. By training on a wide variety of simulated environments, the policy learns to adapt to parameter variations.

```python
class DomainRandomization:
    def __init__(self, env, randomization_params):
        self.env = env
        self.randomization_params = randomization_params

    def randomize_environment(self):
        """
        Randomize environment parameters within specified ranges
        """
        # Randomize physical properties
        for param_name, (min_val, max_val) in self.randomization_params.items():
            random_value = np.random.uniform(min_val, max_val)
            self.env.set_parameter(param_name, random_value)

    def train_with_randomization(self, policy, num_episodes):
        """
        Train policy with domain randomization
        """
        for episode in range(num_episodes):
            # Randomize environment at the start of each episode
            self.randomize_environment()

            # Train on randomized environment
            state = self.env.reset()
            episode_reward = 0

            for step in range(self.env.max_steps):
                action = policy.select_action(state)
                next_state, reward, done, info = self.env.step(action)

                # Update policy
                policy.update(state, action, reward, next_state, done)

                state = next_state
                episode_reward += reward

                if done:
                    break
```

### Physical Parameter Randomization
```python
# Example randomization ranges for humanoid robot
PHYSICAL_RANDOMIZATION_RANGES = {
    # Mass properties
    'robot_mass': [0.8, 1.2],  # Scale factor for robot mass
    'link_mass': [0.7, 1.3],   # Scale factor for individual link masses
    'inertia': [0.8, 1.2],     # Scale factor for inertia tensors

    # Friction parameters
    'ground_friction': [0.4, 1.6],
    'joint_friction': [0.5, 1.5],
    'damping': [0.7, 1.3],

    # Actuator parameters
    'max_torque': [0.8, 1.2],
    'gear_ratio': [0.95, 1.05],
    'motor_constant': [0.9, 1.1],

    # Sensor parameters
    'imu_noise': [0.5, 2.0],   # Scale factor for IMU noise
    'joint_encoder_noise': [0.8, 1.2],

    # Environmental parameters
    'gravity': [9.5, 10.1],    # m/s^2
    'wind_force': [0.0, 0.5],  # Random horizontal forces
}

def apply_physical_randomization(env, randomization_ranges):
    """
    Apply randomization to physical parameters of the environment
    """
    for param, (min_val, max_val) in randomization_ranges.items():
        if param == 'robot_mass':
            scale_factor = np.random.uniform(min_val, max_val)
            env.set_robot_mass_scale(scale_factor)
        elif param == 'ground_friction':
            friction = np.random.uniform(min_val, max_val)
            env.set_ground_friction(friction)
        elif param == 'gravity':
            gravity = np.random.uniform(min_val, max_val)
            env.set_gravity([0, 0, -gravity])
        # Add more parameter randomizations as needed
```

### Visual Domain Randomization
For vision-based humanoid robots, visual domain randomization randomizes visual properties:

```python
class VisualDomainRandomization:
    def __init__(self, env):
        self.env = env
        self.visual_params = {
            'lighting': {
                'intensity_range': [0.5, 2.0],
                'direction_range': [0, 2*np.pi],
                'color_temperature_range': [3000, 8000]
            },
            'texture': {
                'random_textures': True,
                'color_variations': True,
                'material_properties': True
            },
            'camera': {
                'noise_level_range': [0.0, 0.1],
                'blur_range': [0.0, 2.0],
                'brightness_range': [0.8, 1.2],
                'contrast_range': [0.8, 1.2]
            }
        }

    def randomize_visuals(self):
        """
        Randomize visual properties of the environment
        """
        # Randomize lighting
        intensity = np.random.uniform(*self.visual_params['lighting']['intensity_range'])
        direction = np.random.uniform(*self.visual_params['lighting']['direction_range'])
        self.env.set_lighting(intensity, direction)

        # Randomize textures and materials
        if self.visual_params['texture']['random_textures']:
            self.env.apply_random_textures()

        # Randomize camera properties
        noise_level = np.random.uniform(*self.visual_params['camera']['noise_level_range'])
        self.env.set_camera_noise(noise_level)
```

## System Identification

### Parameter Estimation
System identification involves estimating the true parameters of the real robot to improve simulation accuracy:

```python
class SystemIdentifier:
    def __init__(self, robot, sim_env):
        self.robot = robot
        self.sim_env = sim_env
        self.parameters = {}
        self.identified_params = {}

    def excite_system(self, input_signal):
        """
        Excite the system with a known input signal to collect data
        """
        # Apply input signal to real robot
        real_data = self.collect_robot_data(input_signal)

        # Apply same input to simulation
        sim_data = self.collect_sim_data(input_signal)

        return real_data, sim_data

    def estimate_parameters(self, real_data, sim_data):
        """
        Estimate system parameters by minimizing the difference between real and simulated behavior
        """
        def objective(params):
            # Update simulation with current parameter estimate
            self.sim_env.update_parameters(params)

            # Simulate with new parameters
            sim_output = self.sim_env.simulate(real_data['input'])

            # Calculate error between real and simulated outputs
            error = np.mean((real_data['output'] - sim_output) ** 2)
            return error

        # Initial parameter guess
        initial_params = self.sim_env.get_parameters()

        # Optimize parameters
        result = scipy.optimize.minimize(objective, initial_params, method='BFGS')

        self.identified_params = result.x
        return self.identified_params

    def collect_robot_data(self, input_signal):
        """
        Collect data from the real robot
        """
        # Reset robot to initial state
        self.robot.reset()

        # Apply input signal and record responses
        states = []
        actions = []
        outputs = []

        for t, input_val in enumerate(input_signal):
            state = self.robot.get_state()
            action = input_val
            output = self.robot.apply_action(action)

            states.append(state)
            actions.append(action)
            outputs.append(output)

        return {
            'input': input_signal,
            'state': np.array(states),
            'action': np.array(actions),
            'output': np.array(outputs)
        }
```

### Bayesian System Identification
```python
class BayesianSystemIdentifier:
    def __init__(self, prior_params, noise_model):
        self.prior_params = prior_params
        self.noise_model = noise_model
        self.posterior_samples = []

    def update_posterior(self, real_data, sim_data):
        """
        Update parameter posterior distribution using Bayesian inference
        """
        # Define likelihood function
        def likelihood(params):
            # Simulate with current parameters
            sim_output = self.simulate_with_params(sim_data['input'], params)

            # Calculate likelihood of observing real data given sim output
            residuals = real_data['output'] - sim_output
            log_likelihood = np.sum(self.noise_model.log_pdf(residuals))
            return log_likelihood

        # Use MCMC to sample from posterior
        sampler = self.setup_mcmc(likelihood)
        samples = sampler.sample(num_samples=1000)

        self.posterior_samples = samples
        return samples

    def get_parameter_estimate(self):
        """
        Get parameter estimate from posterior samples
        """
        mean_params = np.mean(self.posterior_samples, axis=0)
        std_params = np.std(self.posterior_samples, axis=0)

        return {
            'mean': mean_params,
            'std': std_params,
            'samples': self.posterior_samples
        }
```

## Domain Adaptation Techniques

### Adversarial Domain Adaptation
Adversarial domain adaptation uses a discriminator to distinguish between simulation and real data, forcing the policy to produce indistinguishable behavior:

```python
import torch
import torch.nn as nn

class AdversarialDomainAdapter:
    def __init__(self, policy_network, state_dim):
        self.policy = policy_network
        self.discriminator = self.build_discriminator(state_dim)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=3e-4)

    def build_discriminator(self, state_dim):
        """
        Build discriminator network to distinguish sim vs. real states
        """
        return nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def train_adversarial(self, sim_buffer, real_buffer, num_steps):
        """
        Train with adversarial domain adaptation
        """
        for step in range(num_steps):
            # Sample from both sim and real buffers
            sim_states = sim_buffer.sample(batch_size)
            real_states = real_buffer.sample(batch_size)

            # Train discriminator
            real_labels = torch.ones(batch_size, 1)
            sim_labels = torch.zeros(batch_size, 1)

            # Discriminator loss
            real_output = self.discriminator(real_states)
            sim_output = self.discriminator(sim_states)

            disc_real_loss = nn.BCELoss()(real_output, real_labels)
            disc_sim_loss = nn.BCELoss()(sim_output, sim_labels)
            disc_loss = disc_real_loss + disc_sim_loss

            self.discriminator_optimizer.zero_grad()
            disc_loss.backward()
            self.discriminator_optimizer.step()

            # Train policy to fool discriminator (domain confusion)
            sim_states_for_policy = sim_buffer.sample(batch_size)
            sim_output = self.discriminator(sim_states_for_policy)

            # Policy should generate states that discriminator thinks are "real"
            domain_confusion_loss = nn.BCELoss()(sim_output, real_labels)

            # Combine with original RL objective
            rl_loss = self.compute_rl_loss()  # Standard RL loss
            total_loss = rl_loss + 0.1 * domain_confusion_loss  # Weighted combination

            self.policy_optimizer.zero_grad()
            total_loss.backward()
            self.policy_optimizer.step()
```

### Causal Domain Adaptation
```python
class CausalDomainAdapter:
    def __init__(self, policy_network):
        self.policy = policy_network
        self.causal_graph = self.build_causal_graph()

    def build_causal_graph(self):
        """
        Build causal graph representing relationships between variables
        """
        # Define causal relationships in humanoid control
        causal_graph = {
            'action': ['state', 'policy_output'],
            'state': ['dynamics', 'sensors', 'environment'],
            'dynamics': ['mass', 'friction', 'gravity'],
            'sensors': ['noise', 'bias', 'latency'],
            'environment': ['terrain', 'obstacles', 'disturbances']
        }
        return causal_graph

    def adapt_policy(self, sim_env, real_env):
        """
        Adapt policy using causal relationships
        """
        # Identify causal variables that differ between sim and real
        causal_differences = self.identify_causal_differences(sim_env, real_env)

        # Adjust policy to account for causal differences
        adapted_policy = self.modify_policy_for_causal_differences(
            self.policy, causal_differences
        )

        return adapted_policy
```

## Robust Control Design

### H-infinity Control
H-infinity control designs controllers that are robust to model uncertainties:

```python
def design_hinfinity_controller(plant_model, weight_functions):
    """
    Design H-infinity controller for robust performance
    """
    # Define weighting functions for performance and robustness
    W1 = weight_functions['sensitivity']      # For sensitivity function
    W2 = weight_functions['control']          # For control effort
    W3 = weight_functions['complementary']    # For complementary sensitivity

    # Formulate the augmented plant
    P_augmented = augment_plant(plant_model, W1, W2, W3)

    # Synthesize H-infinity controller
    K = hinfinity_synthesis(P_augmented)

    return K

class RobustHumanoidController:
    def __init__(self, nominal_model, uncertainty_bounds):
        self.nominal_model = nominal_model
        self.uncertainty_bounds = uncertainty_bounds
        self.robust_controller = self.design_robust_controller()

    def design_robust_controller(self):
        """
        Design robust controller using H-infinity synthesis
        """
        # Define uncertainty model
        uncertainty_model = self.build_uncertainty_model()

        # Augment nominal model with uncertainty
        augmented_model = augment_with_uncertainty(
            self.nominal_model, uncertainty_model
        )

        # Design H-infinity controller
        K = design_hinfinity_controller(
            augmented_model,
            self.get_weight_functions()
        )

        return K

    def build_uncertainty_model(self):
        """
        Build uncertainty model based on sim-to-real differences
        """
        # Model uncertainty as multiplicative or additive uncertainty
        uncertainty = {
            'mass_uncertainty': self.uncertainty_bounds['mass'],
            'damping_uncertainty': self.uncertainty_bounds['damping'],
            'friction_uncertainty': self.uncertainty_bounds['friction'],
            'sensor_noise': self.uncertainty_bounds['sensor']
        }

        return uncertainty
```

### Adaptive Control
Adaptive control adjusts controller parameters online based on observed performance:

```python
class AdaptiveHumanoidController:
    def __init__(self, initial_params, adaptation_rate=0.01):
        self.params = initial_params
        self.adaptation_rate = adaptation_rate
        self.reference_model = self.build_reference_model()

    def update_parameters(self, state_error, control_error):
        """
        Update controller parameters based on tracking error
        """
        # Calculate parameter update using gradient descent
        param_gradient = self.calculate_param_gradient(state_error, control_error)
        self.params += self.adaptation_rate * param_gradient

        # Ensure parameters stay within safe bounds
        self.params = np.clip(self.params, self.param_bounds[0], self.param_bounds[1])

    def calculate_param_gradient(self, state_error, control_error):
        """
        Calculate gradient of cost function with respect to parameters
        """
        # Use least squares or other methods to estimate gradient
        gradient = -2 * state_error * self.param_sensitivity_matrix
        return gradient

    def control_step(self, state, reference):
        """
        Perform one control step with adaptive parameters
        """
        # Generate control action using current parameters
        control_action = self.compute_control_action(state, reference)

        # Update parameters based on tracking performance
        state_error = reference - state
        self.update_parameters(state_error, control_action)

        return control_action
```

## Transfer Learning Strategies

### Fine-Tuning Approaches
```python
class SimToRealTransferLearner:
    def __init__(self, pretrained_policy):
        self.pretrained_policy = pretrained_policy
        self.real_env_policy = self.initialize_real_policy(pretrained_policy)

    def initialize_real_policy(self, pretrained_policy):
        """
        Initialize real-world policy based on pretrained simulation policy
        """
        # Copy policy weights from simulation
        real_policy = type(pretrained_policy)()
        real_policy.load_state_dict(pretrained_policy.state_dict())

        # Add adaptation layers for sim-to-real differences
        real_policy.add_adaptation_layers()

        return real_policy

    def transfer_learning(self, real_env, num_real_episodes=100):
        """
        Perform transfer learning from simulation to reality
        """
        # Collect initial data from real environment
        initial_real_data = self.collect_initial_data(real_env)

        # Fine-tune policy on real data
        for episode in range(num_real_episodes):
            state = real_env.reset()

            # Use policy with gradual increase in autonomy
            for step in range(real_env.max_steps):
                # Initially rely more on simulation policy, gradually increase real policy
                interpolation_factor = min(1.0, episode / (num_real_episodes * 0.5))

                sim_action = self.pretrained_policy.select_action(state)
                real_action = self.real_env_policy.select_action(state)

                # Blend actions based on training progress
                blended_action = (1 - interpolation_factor) * sim_action + \
                                interpolation_factor * real_action

                next_state, reward, done, info = real_env.step(blended_action)

                # Update real policy
                self.real_env_policy.update(state, blended_action, reward, next_state, done)

                state = next_state
                if done:
                    break
```

### Meta-Learning for Rapid Adaptation
```python
class MetaLearningAdapter:
    def __init__(self, meta_learner, base_policy):
        self.meta_learner = meta_learner
        self.base_policy = base_policy

    def adapt_to_new_environment(self, new_env, adaptation_steps=10):
        """
        Adapt policy to new environment using meta-learning
        """
        # Collect small amount of data in new environment
        adaptation_data = self.collect_adaptation_data(new_env, adaptation_steps)

        # Use meta-learner to quickly adapt policy
        adapted_policy = self.meta_learner.adapt(
            self.base_policy, adaptation_data
        )

        return adapted_policy

    def collect_adaptation_data(self, env, num_steps):
        """
        Collect data for rapid adaptation
        """
        # Use exploration strategy optimized for system identification
        data = []
        state = env.reset()

        for step in range(num_steps):
            # Use exploratory actions to identify system properties
            action = self.get_exploratory_action(state, step)
            next_state, reward, done, info = env.step(action)

            data.append((state, action, reward, next_state))

            state = next_state
            if done:
                state = env.reset()

        return data
```

## Validation and Testing

### Simulation Fidelity Assessment
```python
def assess_simulation_fidelity(real_robot, sim_env, test_trajectories):
    """
    Assess how well simulation matches reality
    """
    fidelity_metrics = {}

    for trajectory in test_trajectories:
        # Execute trajectory on real robot
        real_states = execute_trajectory(real_robot, trajectory)

        # Simulate same trajectory
        sim_states = simulate_trajectory(sim_env, trajectory)

        # Calculate fidelity metrics
        position_error = np.mean(np.abs(real_states['pos'] - sim_states['pos']))
        velocity_error = np.mean(np.abs(real_states['vel'] - sim_states['vel']))
        torque_error = np.mean(np.abs(real_states['torque'] - sim_states['torque']))

        fidelity_metrics[trajectory.name] = {
            'position_error': position_error,
            'velocity_error': velocity_error,
            'torque_error': torque_error,
            'overall_fidelity': 1.0 / (1.0 + position_error + velocity_error + torque_error)
        }

    return fidelity_metrics

def calculate_transport_distance(sim_states, real_states):
    """
    Calculate Wasserstein distance between sim and real state distributions
    """
    from scipy.stats import wasserstein_distance

    # Calculate distance for each state dimension
    distances = []
    for dim in range(sim_states.shape[1]):
        dist = wasserstein_distance(
            sim_states[:, dim],
            real_states[:, dim]
        )
        distances.append(dist)

    return np.mean(distances)
```

### Progressive Domain Randomization
```python
class ProgressiveDomainRandomization:
    def __init__(self, env, initial_ranges, final_ranges, schedule):
        self.env = env
        self.initial_ranges = initial_ranges
        self.final_ranges = final_ranges
        self.schedule = schedule  # Function of training progress -> parameter ranges
        self.current_ranges = initial_ranges

    def update_randomization(self, training_progress):
        """
        Update randomization ranges based on training progress
        """
        # Interpolate between initial and final ranges based on progress
        alpha = self.schedule(training_progress)

        updated_ranges = {}
        for param, (init_min, init_max) in self.initial_ranges.items():
            final_min, final_max = self.final_ranges[param]

            current_min = init_min + alpha * (final_min - init_min)
            current_max = init_max + alpha * (final_max - init_max)

            updated_ranges[param] = (current_min, current_max)

        self.current_ranges = updated_ranges
        return updated_ranges

    def randomize_environment(self, training_progress):
        """
        Randomize environment with updated ranges
        """
        self.update_randomization(training_progress)

        for param, (min_val, max_val) in self.current_ranges.items():
            random_value = np.random.uniform(min_val, max_val)
            self.env.set_parameter(param, random_value)
```

## Practical Implementation Guidelines

### Best Practices for Sim-to-Real Transfer
1. **Start Simple**: Begin with basic behaviors and gradually increase complexity
2. **Validate Simulation**: Ensure simulation accurately models the real system
3. **Use Multiple Simulations**: Train with various simulation conditions
4. **Monitor Performance**: Continuously assess real-world performance
5. **Safety First**: Implement safety measures for real robot deployment

### Troubleshooting Common Issues
- **Poor Transfer**: Increase domain randomization or collect more real data
- **Oscillatory Behavior**: Reduce controller gains or add more damping
- **Parameter Drift**: Implement parameter bounds and regularization
- **Safety Violations**: Add safety filters and emergency stops

## NVIDIA Isaac™ Integration

### Isaac Sim for Physics Simulation
NVIDIA Isaac Sim provides high-fidelity physics simulation that can bridge the sim-to-real gap:

```python
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage

class IsaacSimToRealTransfer:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.setup_environment()

    def setup_environment(self):
        """
        Set up Isaac Sim environment for sim-to-real transfer
        """
        # Add humanoid robot to stage
        add_reference_to_stage(
            usd_path="/path/to/humanoid_robot.usd",
            prim_path="/World/Humanoid"
        )

        # Configure physics settings for realism
        self.world.scene.enable_collisions = True
        self.world.stage.SetMetadata("upAxis", "Z")

        # Add realistic sensors
        self.add_realistic_sensors()

    def add_realistic_sensors(self):
        """
        Add sensors with realistic noise and latency
        """
        # IMU with noise parameters matching real sensor
        self.imu = self.world.scene.add(
            Imu(
                prim_path="/World/Humanoid/IMU",
                frequency=100,
                noise_density=1.5e-4,  # From real IMU specs
                random_walk=1.5e-6    # From real IMU specs
            )
        )

        # Joint position sensors with encoder noise
        self.joint_sensors = []
        for joint_name in self.humanoid.joint_names:
            sensor = self.world.scene.add(
                JointPositionSensor(
                    prim_path=f"/World/Humanoid/{joint_name}_sensor",
                    joint_name=joint_name,
                    noise_mean=0.0,
                    noise_std=0.001  # Typical encoder noise
                )
            )
            self.joint_sensors.append(sensor)
```

## Summary

Simulation-to-reality transfer is a complex but essential aspect of humanoid robot development. Success requires a combination of techniques including domain randomization, system identification, robust control design, and careful validation. The key is to understand the specific challenges of your humanoid platform and apply appropriate techniques to bridge the reality gap. With proper implementation, sim-to-real transfer can significantly accelerate development while maintaining safety and reducing costs.
