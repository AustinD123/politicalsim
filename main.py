from model import PoliticalGymEnvironment
from callbacks import ElectionRewardCallback
def train_with_stable_baselines():
    """Simple training example"""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    # Create environment
    def make_env():
        return PoliticalGymEnvironment(num_voters=50, num_politicians=3,policy_dimensions=5)  # Smaller for faster training
    
    env = DummyVecEnv([make_env])
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        tensorboard_log="./political_tensorboard/"
    )
    
    # Train
    
    print("Starting training...")
    model.learn(total_timesteps=50000, callback=ElectionRewardCallback())

    
    # Save model
    model.save("political_agents")
    
    # Test trained model
    obs = env.reset()
    for step in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        
        if step % 20 == 0:
            print(f"Step {step}, Rewards: {rewards[0]}")
            env.envs[0].render()
    
    env.close()

# Usage example
if __name__ == "__main__":
    train_with_stable_baselines()
