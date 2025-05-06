import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack, TimeLimit
import cv2
from collections import deque
import torch
import numpy as np
from train import QNet, NoisyLinear
    
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

class Agent(object):
    def __init__(self):
        self.skip = 4
        self.stack = 4
        self.resize = (84, 84)
        self.frame_deque = deque(maxlen=self.stack)
        self.action_space = gym.spaces.Discrete(12)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = QNet(state_size=self.stack , action_size=self.action_space.n).to(self.device)
        self.net.load_state_dict(torch.load("checkpoints/mario_dqn12_ep390.pth"))
        self.net.eval()
        # self.action_step = -1
        # self.action = self.action_space.sample()
        self.action_list = []


    def act(self, observation):
        # self.action_step += 1
        if self.action_list:
            return self.action_list.pop()
     
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        resize = cv2.resize(gray, self.resize, interpolation=cv2.INTER_AREA)
        self.frame_deque.append(resize)
        while len(self.frame_deque) < self.stack:
            self.frame_deque.append(resize)
        stack = np.stack(self.frame_deque, axis=0)
        state = torch.tensor(stack, dtype=torch.float32).to(self.device).unsqueeze(0)
    
        # if self.action_step % self.skip == 0:
        #     with torch.no_grad():
        #         q = self.net(state)
            
        #     self.action = q.argmax(dim=1).item()
        #     return self.action
        # else:
        #     return self.action

        with torch.no_grad():
            q = self.net(state)

            for _ in range(4):
                self.action_list.append(q.argmax(dim=1).item())
        return self.action_list.pop()
    


if __name__ == "__main__":
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    # env = SkipFrame(env, skip=4)
    # env = GrayScaleObservation(env)
    # env = ResizeObservation(env, shape=84)
    # env = FrameStack(env, num_stack=4)
    # env = TimeLimit(env, max_episode_steps=3000)
    agent = Agent()
    reward_history = []
    frames = []

    for episode in range(1):
        agent = Agent()
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            state, reward, done, info = env.step(action)
            total_reward += reward
            # if episode == 0:
            #     frames.append(env.render(mode='rgb_array'))
            # env.render()

        reward_history.append(total_reward)
        print(f"Episode {episode} Reward: {total_reward}")

    env.close()
    avg_reward = np.mean(reward_history)
    print(f"Average reward over 10 episodes: {avg_reward}")

