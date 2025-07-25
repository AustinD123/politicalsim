# political_simulation/environment.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from mesa import Agent, Model
from mesa.datacollection import DataCollector
from typing import Dict, List, Tuple
import random

class VoterAgent(Agent):
    """Individual voter with multi-dimensional policy preferences"""
    
    def __init__(self, unique_id: int, model, policy_dimensions: int = 8):
        super().__init__(unique_id, model)
        
        # Policy preferences: [healthcare, education, environment, economy, defense, immigration, taxes, social]
        self.policy_preferences = np.random.normal(0, 1, policy_dimensions)
        self.policy_preferences = np.clip(self.policy_preferences, -2, 2)  # Normalize to [-2, 2]
        
        # Voter characteristics
        self.voting_weight = 1.0  # Could represent turnout probability
        self.susceptibility_to_change = np.random.beta(2, 5)  # How easily preferences change
        self.social_influence_weight = np.random.uniform(0.1, 0.3)
        
        # Historical voting record
        self.voting_history = []
        self.satisfaction_history = []
    
    def calculate_policy_alignment(self, politician_policy: np.ndarray) -> float:
        """Calculate alignment with politician using cosine similarity"""
        dot_product = np.dot(self.policy_preferences, politician_policy)
        norms = np.linalg.norm(self.policy_preferences) * np.linalg.norm(politician_policy)
        if norms == 0:
            return 0
        return dot_product / norms
    
    def update_preferences(self, external_shock: Dict = None, social_influence: np.ndarray = None):
        """Update preferences based on external events and social influence"""
        if external_shock:
            # Apply external shock (e.g., economic crisis affects economy preference)
            for dimension, change in external_shock.items():
                if dimension < len(self.policy_preferences):
                    self.policy_preferences[dimension] += change * self.susceptibility_to_change
        
        if social_influence is not None:
            # Social influence from neighboring voters
            influence = social_influence * self.social_influence_weight * self.susceptibility_to_change
            self.policy_preferences += influence
        
        # Keep preferences within bounds
        self.policy_preferences = np.clip(self.policy_preferences, -2, 2)
    
    def vote(self, politicians: List['PoliticianAgent']) -> int:
        """Vote for politician with highest policy alignment"""
        alignments = [self.calculate_policy_alignment(p.current_policy) for p in politicians]
        chosen_politician = np.argmax(alignments)
        
        self.voting_history.append(chosen_politician)
        self.satisfaction_history.append(max(alignments))
        
        return chosen_politician

class PoliticianAgent:
    """RL-based politician agent that learns optimal policies"""
    
    def __init__(self, agent_id: int, policy_dimensions: int = 8):
        self.agent_id = agent_id
        self.policy_dimensions = policy_dimensions
        
        # Initialize random policy
        self.current_policy = np.random.normal(0, 0.5, policy_dimensions)
        self.current_policy = np.clip(self.current_policy, -2, 2)
        
        # RL-related attributes
        self.vote_history = []
        self.reward_history = []
        self.policy_history = []
    
    def update_policy(self, action: np.ndarray):
        """Update policy based on RL action"""
        # Action represents policy adjustments
        self.current_policy += action * 0.1  # Small incremental changes
        self.current_policy = np.clip(self.current_policy, -2, 2)
        self.policy_history.append(self.current_policy.copy())

class PoliticalEnvironment(gym.Env):
    """Gymnasium environment for political simulation"""
    
    def __init__(self, num_voters: int = 100, num_politicians: int = 5, policy_dimensions: int = 8):
        super().__init__()
        
        self.num_voters = num_voters
        self.num_politicians = num_politicians
        self.policy_dimensions = policy_dimensions
        
        # Action space: each politician can adjust their policy in each dimension
        # Continuous actions between -0.2 and 0.2 for policy adjustments
        self.action_space = spaces.Box(
            low=-0.2, high=0.2, 
            shape=(num_politicians, policy_dimensions), 
            dtype=np.float32
        )
        
        # Observation space: voter preferences + politician policies + vote counts
        obs_size = (num_voters * policy_dimensions +  # voter preferences
                   num_politicians * policy_dimensions +  # politician policies
                   num_politicians)  # vote counts from last election
        
        self.observation_space = spaces.Box(
            low=-3, high=3, shape=(obs_size,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None):
        """Reset the environment"""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Initialize voters
        self.voters = [VoterAgent(i, None, self.policy_dimensions) for i in range(self.num_voters)]
        
        # Initialize politicians
        self.politicians = [PoliticianAgent(i, self.policy_dimensions) for i in range(self.num_politicians)]
        
        self.current_step = 0
        self.vote_counts = np.zeros(self.num_politicians)
        
        return self._get_observation(), {}
    
    def step(self, actions):
        """Execute one step in the environment"""
        # Update politician policies based on actions
        for i, politician in enumerate(self.politicians):
            politician.update_policy(actions[i])
        
        # Apply external shocks every 10 steps
        if self.current_step % 10 == 0 and self.current_step > 0:
            self._apply_external_shock()
        
        # Conduct election
        votes = []
        for voter in self.voters:
            vote = voter.vote(self.politicians)
            votes.append(vote)
        
        # Count votes
        self.vote_counts = np.bincount(votes, minlength=self.num_politicians)
        
        # Calculate rewards (votes received)
        rewards = self.vote_counts.astype(np.float32)
        
        # Update politician histories
        for i, politician in enumerate(self.politicians):
            politician.vote_history.append(self.vote_counts[i])
            politician.reward_history.append(rewards[i])
        
        self.current_step += 1
        
        # Check if done (arbitrary stopping condition)
        done = self.current_step >= 1000
        
        return self._get_observation(), rewards, done, False, {}
    
    def _apply_external_shock(self):
        """Apply external shock to voter preferences"""
        shock_types = [
            {'type': 'economic_crisis', 'dimensions': {3: -0.5}},  # Economy dimension
            {'type': 'pandemic', 'dimensions': {0: 0.3, 2: 0.2}},  # Healthcare, Environment
            {'type': 'war', 'dimensions': {4: 0.4, 3: -0.2}},  # Defense up, Economy down
        ]
        
        selected_shock = random.choice(shock_types)
        
        for voter in self.voters:
            voter.update_preferences(external_shock=selected_shock['dimensions'])
        
        print(f"Applied {selected_shock['type']} shock at step {self.current_step}")
    
    def _get_observation(self):
        """Get current observation state"""
        obs = []
        
        # Voter preferences
        for voter in self.voters:
            obs.extend(voter.policy_preferences)
        
        # Politician policies
        for politician in self.politicians:
            obs.extend(politician.current_policy)
        
        # Vote counts from last election
        obs.extend(self.vote_counts)
        
        return np.array(obs, dtype=np.float32)
    
    def render(self, mode='human'):
        """Render the environment state"""
        print(f"Step: {self.current_step}")
        print(f"Vote counts: {self.vote_counts}")
        for i, politician in enumerate(self.politicians):
            print(f"Politician {i} policy: {politician.current_policy[:3]}...")  # Show first 3 dimensions

# Usage example
if __name__ == "__main__":
    env = PoliticalEnvironment()
    obs, _ = env.reset()
    
    for step in range(50):
        # Random actions for testing
        actions = env.action_space.sample()
        obs, rewards, done, truncated, info = env.step(actions)
        
        if step % 10 == 0:
            env.render()
        
        if done:
            break