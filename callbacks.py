from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter

class ElectionRewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(ElectionRewardCallback, self).__init__(verbose)
        self.writer = None
        self.step = 0

    def _on_training_start(self):
        log_dir = os.path.join(self.logger.dir, "custom")
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

    def _on_step(self):
        rewards = self.locals.get("rewards")
        if rewards is not None:
            avg_reward = np.mean(rewards)
            self.writer.add_scalar("rewards/average", avg_reward, self.num_timesteps)

            # Log individual politician rewards
            for i, reward in enumerate(rewards):  # rewards is [batch x num_politicians]
                self.writer.add_scalar(f"rewards/politician_{i}", reward, self.num_timesteps)
        
        self.step += 1
        return True

    def _on_training_end(self):
        self.writer.close()
