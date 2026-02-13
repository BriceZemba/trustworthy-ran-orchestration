"""PPO Training loop."""
class PPOTrainer:
    """Trainer for PPO algorithm."""
    def __init__(self, policy, env, optimizer, config):
        self.policy = policy
        self.env = env
        self.config = config
    
    def train(self, n_steps):
        """Main training loop."""
        pass
