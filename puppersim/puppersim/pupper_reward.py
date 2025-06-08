import numpy as np

class PupperReward:
    def __init__(self, 
                 forward_weight=1.0,
                 energy_weight=0.1,
                 stability_weight=0.5,
                 smoothness_weight=0.2):
        self.forward_weight = forward_weight
        self.energy_weight = energy_weight
        self.stability_weight = stability_weight
        self.smoothness_weight = smoothness_weight
        
        # Target values for stability
        self.target_height = 0.15  # meters
        self.target_pitch = 0.0    # radians
        self.target_roll = 0.0     # radians
        
    def compute_reward(self, state, action, next_state):
        """Compute the reward for the current state transition."""
        
        # Forward progress reward
        forward_reward = self._compute_forward_reward(state, next_state)
        
        # Energy efficiency reward
        energy_reward = self._compute_energy_reward(action)
        
        # Stability reward
        stability_reward = self._compute_stability_reward(next_state)
        
        # Smoothness reward
        smoothness_reward = self._compute_smoothness_reward(state, next_state)
        
        # Combine rewards
        total_reward = (
            self.forward_weight * forward_reward +
            self.energy_weight * energy_reward +
            self.stability_weight * stability_reward +
            self.smoothness_weight * smoothness_reward
        )
        
        return total_reward
    
    def _compute_forward_reward(self, state, next_state):
        """Reward for forward progress."""
        # Extract x position from state
        current_x = state[0]  # Assuming first element is x position
        next_x = next_state[0]
        
        # Reward is the change in x position
        forward_reward = next_x - current_x
        
        return forward_reward
    
    def _compute_energy_reward(self, action):
        """Penalty for high motor torques."""
        # Penalize squared sum of actions (proportional to energy)
        energy_penalty = -np.sum(action ** 2)
        
        return energy_penalty
    
    def _compute_stability_reward(self, state):
        """Reward for maintaining stable body pose."""
        # Extract relevant state components
        height = state[1]  # Assuming second element is height
        pitch = state[2]   # Assuming third element is pitch
        roll = state[3]    # Assuming fourth element is roll
        
        # Compute height error
        height_error = abs(height - self.target_height)
        height_reward = -height_error
        
        # Compute orientation errors
        pitch_error = abs(pitch - self.target_pitch)
        roll_error = abs(roll - self.target_roll)
        orientation_reward = -(pitch_error + roll_error)
        
        # Combine stability rewards
        stability_reward = height_reward + orientation_reward
        
        return stability_reward
    
    def _compute_smoothness_reward(self, state, next_state):
        """Penalty for jerky movements."""
        # Extract joint angles
        current_joints = state[4:16]  # Assuming joints are elements 4-15
        next_joints = next_state[4:16]
        
        # Compute joint velocity
        joint_velocity = next_joints - current_joints
        
        # Penalize high joint velocities
        smoothness_penalty = -np.sum(joint_velocity ** 2)
        
        return smoothness_penalty 