# agents.py (Corrected again for Mesa 3.0+ Agent.__init__ signature)
import numpy as np
from mesa import Agent
from typing import Dict, List
import random

class VoterAgent(Agent):
    """Simple voter agent that responds to politicians"""
    
    def __init__(self, unique_id: int, model, policy_dimensions: int = 8):
        # CORRECTED: Pass unique_id and model to super().__init__()
        # Mesa's Agent.__init__ still expects these. The confusion was
        # around the automatic ID generation when adding to schedule/AgentSet.
        super().__init__()

        
        # Policy preferences
        self.policy_preferences = np.random.normal(0, 1, policy_dimensions)
        self.policy_preferences = np.clip(self.policy_preferences, -2, 2)
        
        # Simple behavioral parameters
        self.volatility = np.random.uniform(0.01, 0.1)  # How easily preferences change
        self.social_influence_weight = np.random.uniform(0.05, 0.2)
        
        # Memory
        self.voting_history = []
        self.last_satisfaction = 0.0
        self.reward=0.0
        self.unique_id=unique_id
    
    def step(self):
        """Mesa step - apply social influence"""
        self._apply_social_influence()
    
    def _apply_social_influence(self):
        """Simple social influence from neighbors"""
        neighbors = self.model.get_neighbors(self.unique_id)
        if not neighbors:
            return
            
        # Average neighbor preferences
        neighbor_prefs = []
        for neighbor_id in neighbors:
            # Assuming model.voters is a list where index corresponds to unique_id,
            # or you have a reliable way to retrieve agents by ID from the model.
            # A more robust way might be model.get_agent_by_id(neighbor_id) if available,
            # or iterating model.schedule.agents.
            if neighbor_id < len(self.model.voters): # Safety check
                neighbor = self.model.voters[neighbor_id]
                neighbor_prefs.append(neighbor.policy_preferences)
        
        if neighbor_prefs:
            avg_neighbor_pref = np.mean(neighbor_prefs, axis=0)
            influence = self.social_influence_weight * self.volatility
            
            self.policy_preferences = (
                (1 - influence) * self.policy_preferences +
                influence * avg_neighbor_pref
            )
            self.policy_preferences = np.clip(self.policy_preferences, -2, 2)
    
    def calculate_policy_alignment(self, politician_policy: np.ndarray) -> float:
        """Calculate alignment with politician using cosine similarity"""
        norm_self = np.linalg.norm(self.policy_preferences)
        norm_politician = np.linalg.norm(politician_policy)
        
        if norm_self == 0 or norm_politician == 0:
            return 0
        dot_product = np.dot(self.policy_preferences, politician_policy)
        return dot_product / (norm_self * norm_politician)
    
    def vote(self, politicians: List) -> int:
        """Vote for most aligned politician"""
        if not politicians:
            return -1
            
        alignments = [self.calculate_policy_alignment(pol.current_policy) for pol in politicians]
        chosen = np.argmax(alignments)
        
        self.voting_history.append(chosen)
        self.last_satisfaction = max(alignments)
        
        return chosen

class SimplePoliticianAgent(Agent):
    """Simple politician agent - controlled by external RL but with Mesa behaviors"""
    
    def __init__(self, unique_id: int, model, policy_dimensions: int = 8):
        # CORRECTED: Pass unique_id and model to super().__init__()
        super().__init__()
        
        self.policy_dimensions = policy_dimensions
        self.current_policy = np.random.normal(0, 0.5, policy_dimensions)
        self.current_policy = np.clip(self.current_policy, -2, 2)
        
        # History tracking
        self.vote_history = []
        self.policy_history = []
        
        # Simple behavioral traits
        self.consistency_preference = np.random.uniform(0.1, 0.9)  # Resist large policy changes
        self.popularity_seeking = np.random.uniform(0.3, 0.8)     # How much they chase votes
        self.unique_id=unique_id
    
    def step(self):
        """Mesa step - politicians can have autonomous behaviors here"""
        pass
    
    def update_policy(self, action: np.ndarray):
        """Update policy based on RL action (called from Gym environment)"""
        dampened_action = action * (1 - self.consistency_preference)
        
        self.current_policy += dampened_action * 0.1
        self.current_policy = np.clip(self.current_policy, -2, 2)
        self.policy_history.append(self.current_policy.copy())
    
    def get_voter_feedback(self, voters: List[VoterAgent]) -> Dict:
        """Get feedback from voters for this politician"""
        alignments = []
        potential_votes = 0
        
        # Calculate max_possible_distance for normalization, assuming policy range [-2, 2]
        max_possible_distance = np.sqrt(self.policy_dimensions * (2 - (-2))**2)

        for voter in voters:
            distance = np.linalg.norm(voter.policy_preferences - self.current_policy)
            
            normalized_distance = distance / max_possible_distance if max_possible_distance > 0 else 0
            alignment_score = 1 - normalized_distance 

            alignments.append(alignment_score)
            if alignment_score > 0.6:  
                potential_votes += 1
        
        return {
            'avg_alignment': np.mean(alignments) if alignments else 0,
            'potential_supporters': potential_votes,
            'alignment_std': np.std(alignments) if alignments else 0
        }