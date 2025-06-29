'''A swerve environment with three robots navigating to different goals while avoiding collision.'''

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.swerve import binding

class Swerve(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, num_agents=6, render_mode=None, log_interval=128, 
                 max_episode_steps=1000, buf=None, seed=0):
        # Multi-agent environment: each agent has its own observation and action
        # Each robot observes: [goal_dx, goal_dy, goal_angle_diff, vx, vy, omega, other1_dx, other1_dy, other1_dvx, other1_dvy, other2_dx, other2_dy, other2_dvx, other2_dvy]
        # Total observation: 14 dimensions per agent
        self.single_observation_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(14,), dtype=np.float32)
        
        # Action space: [x_vel, y_vel, omega] per agent
        self.single_action_space = gymnasium.spaces.Box(
            low=-2.0, high=2.0, shape=(3,), dtype=np.float32)
        
        self.render_mode = render_mode
        self.num_agents = num_envs * num_agents  # Total agents across all environments
        self.log_interval = log_interval
        self.agents_per_env = num_agents

        super().__init__(buf)
        c_envs = []
        for i in range(num_envs):
            c_env = binding.env_init(
                self.observations[i*num_agents:(i+1)*num_agents],
                self.actions[i*num_agents:(i+1)*num_agents],
                self.rewards[i*num_agents:(i+1)*num_agents],
                self.terminals[i*num_agents:(i+1)*num_agents],
                self.truncations[i*num_agents:(i+1)*num_agents],
                seed, max_episode_steps=max_episode_steps, num_agents=num_agents)
            c_envs.append(c_env)

        self.c_envs = binding.vectorize(*c_envs)

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.tick += 1
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        
        info = []
        if self.tick % self.log_interval == 0:
            log = binding.vec_log(self.c_envs)
            if log:
                info.append(log)
            
        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

if __name__ == '__main__':
    N = 512
    num_agents = 3

    env = Swerve(num_envs=N, num_agents=num_agents)
    env.reset()
    steps = 0

    CACHE = 1024
    actions = np.random.uniform(-1.0, 1.0, (CACHE, 3))

    i = 0
    import time
    start = time.time()
    while time.time() - start < 10:
        env.step(actions[i % CACHE])
        steps += env.num_agents
        i += 1

    print('Swerve SPS:', int(steps / (time.time() - start)))
