import numpy as np
import gymnasium as gym
from gymnasium import spaces
from mesa import Agent, Model
from mesa.datacollection import DataCollector
import networkx as nx
from typing import Dict, List, Any
import random


# Assuming 'agents.py' contains VoterAgent and SimplePoliticianAgent
from agents import VoterAgent
from agents import SimplePoliticianAgent

class PoliticalMesaModel(Model):
    """Mesa model that manages voter-politician interactions"""
    
    def __init__(self, num_voters: int = 100, num_politicians: int = 5, 
                     policy_dimensions: int = 8):
        # Initialize the Model superclass first.
        # In Mesa 3.0+, unique_id for agents is automatically assigned,
        # and there's no need to explicitly pass `random` or `time` to the superclass.
        super().__init__() 
        
        self.num_voters = num_voters
        self.num_politicians = num_politicians
        # self.current_step is now automatically managed by Model.steps
        # but for clarity in this specific model's logic, we can keep it
        # or rely on self.steps. For now, I'll keep it as current_step is also
        # used in _get_observation.
        self.current_step = 0 
        
        # Schedulers are deprecated. Agent management is now handled directly by the Model's AgentSet.
        # You no longer explicitly create a scheduler. Agents are added directly to the model.
        
        # Create social network for voters
        self.social_network = nx.watts_strogatz_graph(num_voters, 6, 0.1)
        
        # Create agents
        self.voters = []
        for i in range(num_voters):
            # In Mesa 3.0+, unique_id is often handled internally.
            # If you need a specific starting ID or want to explicitly set it,
            # you can still pass it, but the simpler way is to let Mesa assign it.
            # For this example, keeping 'i' as the unique_id for clarity
            # and to match existing logic if agents relied on specific IDs.
            voter = VoterAgent(i, self, policy_dimensions) 
            # Agents are automatically added to self.agents when initialized with `model=self`.
            self.voters.append(voter)
        
        self.politicians = []
        for i in range(num_politicians):
            politician = SimplePoliticianAgent(num_voters + i, self, policy_dimensions)
            # Agents are automatically added to self.agents
            self.politicians.append(politician)
        
        # Election tracking
        self.last_election_results = np.zeros(num_politicians)
        
        # Data collection
        self.datacollector = DataCollector(
            model_reporters={
                "Polarization": self.compute_polarization,
                "Avg_Satisfaction": self.compute_avg_satisfaction
            }
        )
    
    def get_neighbors(self, voter_id: int) -> List[int]:
        """Get social network neighbors for a voter"""
        if voter_id < len(self.voters):
            return list(self.social_network.neighbors(voter_id))
        return []
    
    def step(self):
        """Execute one step of the Mesa simulation"""
        # Replace self.schedule.step() with agent.do() or agent.shuffle_do()
        # For random activation (like RandomActivation scheduler), use shuffle_do.
        self.agents.shuffle_do("step")  # All agents take their steps in a random order
        self.datacollector.collect(self)
        self.current_step += 1 # Continue to manually increment if used for specific logic outside Model.steps

    def hold_election(self) -> np.ndarray:
        """Conduct election and return vote counts"""
        vote_counts = np.zeros(self.num_politicians)
        
        for voter in self.voters:
            # Ensure voters are still part of the model's self.voters list if they were created there
            # or iterate over self.agents if all agents should vote.
            # Assuming self.voters list is correctly maintained.
            vote = voter.vote(self.politicians) 
            if vote >= 0:
                vote_counts[vote] += 1
        
        # Update politician histories
        for i, politician in enumerate(self.politicians):
            politician.vote_history.append(vote_counts[i])
        
        self.last_election_results = vote_counts
        return vote_counts
    
    def apply_external_shock(self, shock_type: str = "random"):
        """Apply external shock to voter preferences"""
        shocks = {
            "economic_crisis": {'dimension': 3, 'magnitude': -0.3},
            "pandemic": {'dimension': 0, 'magnitude': 0.2},
            "war": {'dimension': 4, 'magnitude': 0.3}
        }
        
        if shock_type == "random":
            shock_type = random.choice(list(shocks.keys()))
        
        shock = shocks.get(shock_type, {'dimension': 0, 'magnitude': 0.1})
        
        for voter in self.voters:
            change = np.random.normal(shock['magnitude'], 0.1)
            voter.policy_preferences[shock['dimension']] += change * voter.volatility
            voter.policy_preferences = np.clip(voter.policy_preferences, -2, 2)
    
    def compute_polarization(self):
        """Compute political polarization"""
        voter_prefs = np.array([voter.policy_preferences for voter in self.voters])
        center = np.mean(voter_prefs, axis=0)
        distances = [np.linalg.norm(pref - center) for pref in voter_prefs]
        return np.mean(distances)
    
    def compute_avg_satisfaction(self):
        """Compute average voter satisfaction"""
        satisfactions = [voter.last_satisfaction for voter in self.voters]
        return np.mean(satisfactions) if satisfactions else 0

class PoliticalGymEnvironment(gym.Env):
    """Gymnasium environment that wraps the Mesa model"""
    
    def __init__(self, num_voters: int = 100, num_politicians: int = 5, 
                     policy_dimensions: int = 8):
        super().__init__()
        
        self.num_politicians = num_politicians
        self.policy_dimensions = policy_dimensions
        
        # Create Mesa model
        self.mesa_model = PoliticalMesaModel(num_voters, num_politicians, policy_dimensions)
        
        # Gym spaces
        self.action_space = spaces.Box(
            low=-0.2, high=0.2,
            shape=(num_politicians, policy_dimensions),
            dtype=np.float32
        )
        
        # Observation: voter preferences + politician policies + social metrics
        obs_size = (
            num_voters * policy_dimensions +   # voter preferences
            num_politicians * policy_dimensions +   # politician policies   
            num_politicians +   # last vote counts
            3   # polarization, satisfaction, step_count
        )
        
        self.observation_space = spaces.Box(
            low=-5, high=5, shape=(obs_size,), dtype=np.float32
        )
        
        self.current_step = 0
        self.max_steps = 1000
    
    def reset(self, seed=None):
        """Reset the environment"""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Create new Mesa model
        # Re-initializing the model correctly for a reset.
        # No need to pass self.mesa_model.num_voters as an arg directly,
        # it's better to pass the original parameters.
        self.mesa_model = PoliticalMesaModel(
            num_voters=self.mesa_model.num_voters, # Use the original number of voters
            num_politicians=self.num_politicians,
            policy_dimensions=self.policy_dimensions
        )
        
        self.current_step = 0
        return self._get_observation(), {}
    
    def step(self, actions):
        """Execute one step"""
        # Update politician policies based on RL actions
        for i, politician in enumerate(self.mesa_model.politicians):
            politician.update_policy(actions[i])
        
        # Run Mesa simulation step (voter interactions, social influence)
        self.mesa_model.step()
        
        # Apply external shocks periodically
        if self.current_step % 50 == 0 and self.current_step > 0:
            self.mesa_model.apply_external_shock()
        
        # Hold elections periodically
        rewards = np.zeros(self.num_politicians)
        if self.current_step % 20 == 0:
            vote_counts = self.mesa_model.hold_election()
            rewards = vote_counts.astype(np.float32)
        
        # Alternative reward: continuous voter alignment
        else:
            for i, politician in enumerate(self.mesa_model.politicians):
                feedback = politician.get_voter_feedback(self.mesa_model.voters)
                rewards[i] = feedback['avg_alignment'] * 10  # Scale reward
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        return self._get_observation(), rewards, done, False, self._get_info()
    
    def _get_observation(self):
        """Get current observation state"""
        obs = []
        
        # Voter preferences
        for voter in self.mesa_model.voters:
            obs.extend(voter.policy_preferences)
        
        # Politician policies
        for politician in self.mesa_model.politicians:
            obs.extend(politician.current_policy)
        
        # Last election results
        obs.extend(self.mesa_model.last_election_results)
        
        # Social metrics
        obs.extend([
            self.mesa_model.compute_polarization(),
            self.mesa_model.compute_avg_satisfaction(),
            self.current_step / self.max_steps  # Normalized step count
        ])
        
        return np.array(obs, dtype=np.float32)
    
    def _get_info(self):
        """Get additional info"""
        return {
            'polarization': self.mesa_model.compute_polarization(),
            'avg_satisfaction': self.mesa_model.compute_avg_satisfaction(),
            'last_election': self.mesa_model.last_election_results.copy()
        }
    
    def render(self, mode='human'):
        """Render environment state"""
        print(f"Step: {self.current_step}")
        print(f"Polarization: {self.mesa_model.compute_polarization():.3f}")
        print(f"Avg Satisfaction: {self.mesa_model.compute_avg_satisfaction():.3f}")
        print(f"Last Election: {self.mesa_model.last_election_results}")