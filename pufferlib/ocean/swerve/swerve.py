'''A swerve drive environment with two robots navigating to different goals while avoiding collision.'''

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.swerve import binding

class Swerve(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, log_interval=128, max_episode_steps=1000, buf=None, seed=0):
        # Each robot observes: [x, y, theta, vx, vy, omega, goal_x, goal_y, other_robot_x, other_robot_y, other_goal_x, other_goal_y]
        # So observation space is 12 dimensions per robot, 2 robots = 24 total
        self.single_observation_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(24,), dtype=np.float32)
        
        # Action space: [x_vel, y_vel, omega] for each robot = 6 actions total
        self.single_action_space = gymnasium.spaces.Box(
            low=np.array([-2.0, -2.0, -3.14, -2.0, -2.0, -3.14]), 
            high=np.array([2.0, 2.0, 3.14, 2.0, 2.0, 3.14]), 
            dtype=np.float32)
        
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.max_episode_steps = max_episode_steps
        self.log_interval = log_interval

        super().__init__(buf)
        self.c_envs = binding.vec_init(self.observations, self.actions, self.rewards,
            self.terminals, self.truncations, num_envs, seed, max_episode_steps=max_episode_steps)
 
    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.tick += 1
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        
        if self.tick % self.log_interval == 0:
            info = [binding.vec_log(self.c_envs)]
        else:
            info = [{}]
            
        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

if __name__ == '__main__':
    N = 4096
    env = Swerve(num_envs=N)
    env.reset()
    steps = 0

    CACHE = 1024
    actions = np.random.uniform(-1.0, 1.0, (CACHE, N, 6))

    import time
    start = time.time()
    while time.time() - start < 10:
        env.step(actions[steps % CACHE])
        steps += 1

    print('Swerve SPS:', int(env.num_agents*steps / (time.time() - start)))
