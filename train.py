import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack, TimeLimit
import matplotlib.pyplot as plt


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break

        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        return max_frame, total_reward, done, info
    
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
    
    
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=2.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.w_mu = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.w_sigma = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.b_mu = torch.nn.Parameter(torch.empty(out_features))
        self.b_sigma = torch.nn.Parameter(torch.empty(out_features))

        self.register_buffer('w_epsilon', torch.Tensor(out_features, in_features))
        self.register_buffer('b_epsilon', torch.Tensor(out_features))

        self.init_parameters()
        self.reset_noise()

    def init_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.w_mu.data.uniform_(-mu_range, mu_range)
        self.b_mu.data.uniform_(-mu_range, mu_range)
        self.w_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.b_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def _noise_func(self, size):
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    def reset_noise(self):
        in_epsilon = self._noise_func(self.in_features)
        out_epsilon = self._noise_func(self.out_features)
        self.w_epsilon.copy_(out_epsilon.ger(in_epsilon))
        self.b_epsilon.copy_(out_epsilon)
    
    def forward(self, x):
        if self.training:
            weight = self.w_mu + self.w_sigma * self.w_epsilon
            bias = self.b_mu + self.b_sigma * self.b_epsilon
        else:
            weight = self.w_mu
            bias = self.b_mu
        return F.linear(x, weight, bias)

class QNet(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(state_size, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.value_stream = nn.Sequential(
            NoisyLinear(7*7*64, 512),
            nn.ReLU(),
            NoisyLinear(512, 1)
        )
        self.advantage_stream = nn.Sequential(
            NoisyLinear(7*7*64, 512),
            nn.ReLU(),
            NoisyLinear(512, action_size)
        )

    def forward(self, x):
        x = x / 255.0 
        features = self.features(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

# class ReplayBuffer:
#     def __init__(self, capacity):
#         self.buffer = deque(maxlen=capacity)

#     def add(self, state, action, reward, next_state, done):
#         state = torch.tensor(state, dtype=torch.float32)
#         state = state.squeeze(-1)
#         action = torch.tensor(action, dtype=torch.long)
#         reward = torch.tensor(reward, dtype=torch.float32)
#         next_state = torch.tensor(next_state, dtype=torch.float32)
#         next_state = next_state.squeeze(-1)
#         done = torch.tensor(done, dtype=torch.float32)
#         self.buffer.append((state, action, reward, next_state, done))

#     def sample(self, batch_size):
#         batch = random.sample(self.buffer, batch_size)
#         states, actions, rewards, next_states, dones = zip(*batch)
#         states = torch.stack(states)
#         actions = torch.stack(actions)
#         rewards = torch.stack(rewards)
#         next_states = torch.stack(next_states)
#         dones = torch.stack(dones)
#         return states, actions, rewards, next_states, dones

#     def __len__(self):
#         return len(self.buffer)
    
# class NstepReplayBuffer:
#     def __init__(self, capacity, n_step=9, gamma=0.99):
#         self.capacity = capacity
#         self.buffer = deque(maxlen=capacity)
#         self.n_step = n_step
#         self.gamma = gamma
#         self.nstep_buffer = deque(maxlen=n_step)

#     def add(self, state, action, reward, next_state, done):
#         self.nstep_buffer.append((state, action, reward, next_state, done))
#         if len(self.nstep_buffer) < self.n_step:
#             return

#         # 計算 n-step return
#         R, final_next_state, final_done = 0, None, False
#         for idx, (_, _, r, _, d) in enumerate(self.nstep_buffer):
#             R += (self.gamma ** idx) * r
#             if d:
#                 final_done = True
#                 break
#         state, action, _, _, _ = self.nstep_buffer[0]
#         _, _, _, final_next_state, _ = self.nstep_buffer[-1]
#         self.buffer.append((torch.tensor(state, dtype=torch.float32).squeeze(-1),
#                             torch.tensor(action, dtype=torch.long),
#                             torch.tensor(R, dtype=torch.float32),
#                             torch.tensor(final_next_state, dtype=torch.float32).squeeze(-1),
#                             torch.tensor(final_done, dtype=torch.float32)))
#         if final_done:
#             self.nstep_buffer.clear()

#     def sample(self, batch_size):
#         batch = random.sample(self.buffer, batch_size)
#         states, actions, rewards, next_states, dones = zip(*batch)
#         return (torch.stack(states),
#                 torch.stack(actions),
#                 torch.stack(rewards),
#                 torch.stack(next_states),
#                 torch.stack(dones))

#     def __len__(self):
#         return len(self.buffer)


class NstepPrioritizedReplayBuffer:
    def __init__(self, capacity, n_step=3, gamma=0.9,
                 alpha=0.6, beta_start=0.4, beta_frames=2000000):
        self.capacity = capacity
        self.buffer = []        
        self.priorities = []  
        self.pos = 0       

        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1

        self.n_step = n_step
        self.gamma = gamma
        self.n_deque = deque(maxlen=n_step)

    def _get_nstep_info(self):
        R = 0.0
        for idx, (_, _, r, _, _) in enumerate(self.n_deque):
            R += (self.gamma**idx) * r
        next_state, done = self.n_deque[-1][3], self.n_deque[-1][4]
        return R, next_state, done

    def add(self, state, action, reward, next_state, done):
        s = torch.tensor(state, dtype=torch.float32).squeeze(-1)
        a = torch.tensor(action, dtype=torch.long)
        r = torch.tensor(reward, dtype=torch.float32)
        s_next = torch.tensor(next_state, dtype=torch.float32).squeeze(-1)
        d = torch.tensor(done, dtype=torch.float32)

        self.n_deque.append((s, a, r, s_next, d))

        if len(self.n_deque) == self.n_step or done:
            while len(self.n_deque) > 0:
                if len(self.n_deque) < self.n_step and not done:
                    break 

                s0, a0, _, _, _ = self.n_deque[0]
                R, s_n, d_n = self._get_nstep_info()
                data = (s0, a0, R, s_n, d_n)

                max_p = max(self.priorities, default=1.0)
                if len(self.buffer) < self.capacity:
                    self.buffer.append(data)
                    self.priorities.append(max_p)
                else:
                    self.buffer[self.pos] = data
                    self.priorities[self.pos] = max_p
                    self.pos = (self.pos + 1) % self.capacity

                self.n_deque.popleft() 

            if done:
                self.n_deque.clear()


    def sample(self, batch_size):
        N = len(self.buffer)
        probs = np.array(self.priorities, dtype=np.float32) ** self.alpha
        probs /= probs.sum()

        idxs = np.random.choice(N, batch_size, p=probs)

        self.frame += 1
        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)
        weights = (N * probs[idxs]) ** (-beta)
        weights /= weights.max()

        batch = [self.buffer[i] for i in idxs]
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.stack(states),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.float32),
            torch.tensor(idxs, dtype=torch.long),
            torch.tensor(weights, dtype=torch.float32)
        )

    def update_priorities(self, idxs, td_errors):
        for idx, err in zip(idxs, td_errors):
            self.priorities[idx] = abs(err) + 1e-6

    def __len__(self):
        return len(self.buffer)



class DQNAgent:
    def __init__(self, state_size, action_size, device):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = 64
        self.gamma = 0.9
        self.alpha = 0.00025
        self.update_freq = 1
        self.train_step = 0
        self.nstep = 9
        self.device = device
        self.model = QNet(state_size, action_size).to(self.device)
        self.target_model = QNet(state_size, action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.alpha, eps=0.00015)
        # self.memory = ReplayBuffer(500000)
        # self.memory = NstepReplayBuffer(capacity=500000, n_step=self.nstep, gamma=self.gamma)
        self.memory = NstepPrioritizedReplayBuffer(capacity=1000000, n_step=self.nstep)

        self.target_model.eval()


    def get_action(self, state, epsilon=0.1, deterministic=True):
        state_np = np.array(state, dtype=np.float32)
        state = torch.from_numpy(state_np).to(self.device)
        # state = torch.tensor(state, dtype=torch.float32).to(self.device)
        state = state.squeeze(-1)
        if not deterministic and np.random.rand() < epsilon:
            return np.random.randint(self.action_size)
        with torch.no_grad():
            # self.model.reset_noise()
            q_values = self.model(state.unsqueeze(0))
        return torch.argmax(q_values).item()

    def update(self, target, learning=0.0005):
        if target == 'hard':
            self.target_model.load_state_dict(self.model.state_dict())
        elif target == 'soft':
            for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.copy_(learning * param.data + (1.0 - learning) * target_param.data)

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)
    
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)

        with torch.no_grad():
            # self.model.reset_noise()  
            # self.target_model.reset_noise()
            next_actions = self.model(next_states).argmax(dim=1)
            target_q = (self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1))
            targets = rewards + (self.gamma**self.nstep) * target_q * (1 - dones)
        
        # self.model.reset_noise()
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        td_errors = targets - current_q 
        # loss = (F.mse_loss(current_q, targets, reduction='none') * weights).mean()
        loss = (F.smooth_l1_loss(current_q, targets, reduction='none') * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.model.reset_noise()  
        self.target_model.reset_noise()
        self.memory.update_priorities(indices, td_errors.detach().cpu().numpy())
    
        self.train_step += 1
        if self.train_step % self.update_freq == 0:
            self.update(target='soft')


def make_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = SkipFrame(env, skip=4)
    # env = MaxAndSkipEnv(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)
    env = TimeLimit(env, max_episode_steps=3000)
    return env

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env()
    agent = DQNAgent(state_size=env.observation_space.shape[0], action_size=env.action_space.n, device=device)

    num_episodes = 10000
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995
    reward_history = [] 
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.get_action(state, epsilon)
            next_state, reward, done, info = env.step(action)

            agent.memory.add(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            total_reward += reward
            epsilon = max(epsilon_min, epsilon * epsilon_decay)


        print(f"Episode {episode + 1} Reward: {total_reward:.2f} Epsilon: {epsilon:.3f}")
        reward_history.append(total_reward)

        if (episode + 1) % 10 == 0:
            torch.save(agent.model.state_dict(), f"checkpoints/mario_dqn_ep{episode+1}.pth")
            avg_reward = np.mean(reward_history[-10:])
            print(f"Episode {episode + 1}/{num_episodes}, Avg Reward: {avg_reward:.4f}")


    env.close()
    plt.plot(reward_history)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training History")
    plt.savefig("training_reward.png") 
    plt.close()   

if __name__ == "__main__":
    main()
