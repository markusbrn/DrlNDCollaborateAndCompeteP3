import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-2              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPGAgent:
    def __init__(self, agent_size, state_size, action_size, random_seed, discount_factor=0.95, tau=0.02):
        super(MADDPGAgent, self).__init__()
        
        self.agent_size = agent_size
        self.state_size = state_size
        self.action_size = action_size
        self.full_state_size = agent_size*state_size
        self.full_action_size = agent_size*action_size
        
        self.maddpg_agent = []
        for _ in range(agent_size):
            self.maddpg_agent.append(DDPGAgent(self.agent_size, self.state_size, self.action_size, random_seed))
        
        self.discount_factor = GAMMA
        self.tau = TAU

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.t_step = 0
        
        self.noise = True
        self.tstop_noise = 100000


    def reset(self):
        for agent in self.maddpg_agent:
            agent.reset()


    def act(self, obs_all_agents):
        """get actions from all agents in the MADDPG object"""
        #print(obs_all_agents)
        self.t_step = self.t_step + 1
        if self.t_step > self.tstop_noise:
            self.noise = False
        actions = [agent.act(obs, self.noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return np.array(actions).reshape(1,-1)


    def step(self, all_states, all_actions, all_rewards, all_next_states, all_dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(all_states, all_actions, all_rewards, all_next_states, all_dones)
        if self.t_step % UPDATE_EVERY == 0:        
            # Learn, if enough samples are available in memory
            if len(self.memory) > BATCH_SIZE:
                self.learn()


    def learn(self):
        for i, agent in enumerate(self.maddpg_agent):
            all_states, all_actions, all_rewards, all_next_states, all_dones = self.memory.sample()
            
            all_actions_next = []
            all_actions_pred = []
            for i_, agent_ in enumerate(self.maddpg_agent):
                id_ = torch.tensor([i_]).to(device)
                all_actions_next.append(agent_.actor_target(all_next_states.index_select(1,id_).squeeze()))
                all_actions_pred.append(agent_.actor_local(all_states.index_select(1,id_).squeeze()))

            # ---------------------------- update critic ---------------------------- #
            id = torch.tensor([i]).to(device)
            
            # Get predicted next-state actions and Q values from target models
            Q_targets_next = agent.critic_target(all_next_states.reshape(-1,self.full_state_size), torch.cat(all_actions_next, dim=1))
            # Compute Q targets for current states (y_i)
            Q_targets = all_rewards.index_select(1,id) + (self.discount_factor * Q_targets_next * (1 - all_dones.index_select(1,id)))
            # Compute critic loss
            Q_expected = agent.critic_local(all_states.reshape(-1,self.full_state_size), all_actions.squeeze())
            critic_loss = F.mse_loss(Q_expected, Q_targets.detach())
            # Minimize the loss
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            agent.critic_optimizer.step()
            
            # ---------------------------- update actor ---------------------------- #
            # Compute actor loss
            all_actions_pred = [actions if i == id else actions.detach() for i, actions in enumerate(all_actions_pred)]
            
            actor_loss = -agent.critic_local(all_states.reshape(-1,self.full_state_size), torch.cat(all_actions_pred, dim=1)).mean()
            # Minimize the loss
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()

            # ----------------------- update target networks ----------------------- #
            agent.soft_update(agent.critic_local, agent.critic_target, TAU)
            agent.soft_update(agent.actor_local, agent.actor_target, TAU)          



class DDPGAgent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, agent_size, state_size, action_size, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.full_state_size = agent_size*state_size
        self.action_size = action_size
        self.full_action_size = agent_size*action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(self.state_size, self.action_size, random_seed).to(device)
        self.actor_target = Actor(self.state_size, self.action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(self.full_state_size, self.full_action_size, random_seed).to(device)
        self.critic_target = Critic(self.full_state_size, self.full_action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Set weights for local and target actor, respectively, critic the same
        self.hard_copy_weights(self.actor_target, self.actor_local)
        self.hard_copy_weights(self.critic_target, self.critic_local)
        
    def hard_copy_weights(self, target, source):
        """ copy weights from source to target network (part of initialization)"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample() * 0.25
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process."""
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()


    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)


    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.uniform(-1,1) for i in range(len(x))])
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)


    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None],axis=0)).float().to(device)
        actions = torch.from_numpy(np.stack([e.action for e in experiences if e is not None],axis=0)).float().to(device)
        rewards = torch.from_numpy(np.stack([e.reward for e in experiences if e is not None],axis=0)).float().to(device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None],axis=0)).float().to(device)
        dones = torch.from_numpy(np.stack([e.done for e in experiences if e is not None],axis=0).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)


    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
