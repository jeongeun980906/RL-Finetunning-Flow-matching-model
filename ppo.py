device = 'cuda'
from torch.distributions import MultivariateNormal
import torch 
from torch import nn, Tensor


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        self.start_times = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.start_times[:]

class ActorCritic(nn.Module):
    def __init__(self, policy, state_dim,action_std_init):

        super(ActorCritic, self).__init__()
        self.action_var = torch.full((state_dim,), action_std_init * action_std_init).to(device)
        self.actor = policy
        self.state_dim = state_dim
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim+1, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
        
    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.state_dim,), new_action_std * new_action_std).to(device)

    def act(self, state, start_time, end_time):
        action_mean = self.actor.step(state, start_time, end_time)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        t_start = start_time.view(1, 1).expand(state.shape[0], 1)
        inp = torch.cat((state, t_start), dim=1)
        state_val = self.critic(inp)

        return action.detach(), action_logprob.detach(), state_val.detach()
    

    def evaluate(self, state, action, start_time, end_time):
        action_mean = self.actor.step(state, start_time, end_time)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        t_start = start_time.view(1, 1).expand(state.shape[0], 1)
        inp = torch.cat((state, t_start), dim=1)
        state_values = self.critic(inp)
        
        return action_logprobs, state_values, dist_entropy
    


import copy

class PPO:
    def __init__(self, policy, state_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,n_steps=8,action_std_init=0.6):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.n_steps = n_steps
        
        self.buffer = RolloutBuffer()
        
        # copy original weight of policy
        self.original_policy = copy.deepcopy(policy)
        for param in self.original_policy.parameters():
            param.requires_grad = False

        self.policy = ActorCritic(policy,state_dim, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])
        # self.actor_optimizer = torch.optim.Adam(self.policy.actor.parameters(), lr=lr_actor)
        # self.critic_optimizer = torch.optim.Adam(self.policy.critic.parameters(), lr=lr_critic)
        self.policy_old = ActorCritic(policy, state_dim, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        self.action_std = action_std_init
        self.start_actor = False

    def set_action_std(self, new_action_std):
        
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)
        
       
    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")

        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if (self.action_std <= min_action_std):
            self.action_std = min_action_std
            print("setting actor output action_std to min_action_std : ", self.action_std)
        else:
            print("setting actor output action_std to : ", self.action_std)
        self.set_action_std(self.action_std)


        print("--------------------------------------------------------------------------------------------")


    def select_action(self, state, start_time, end_time):

        with torch.no_grad():
            action, action_logprob, state_val = self.policy_old.act(state, start_time, end_time)

        return action.detach().cpu().numpy(), action_logprob.detach().cpu().numpy(), state_val.detach().cpu().numpy()

    def update(self):
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)
        rewards = self.buffer.rewards 
        # discounted_reward = 0
        # for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
        #     if is_terminal:
        #         discounted_reward = 0
        #     discounted_reward = reward + (self.gamma * discounted_reward)
        #     rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        # print(old_states.shape, old_actions.shape, old_logprobs.shape, old_state_values.shape)
        # Optimize policy for K epochs
        time_steps = torch.linspace(0, 1.0, self.n_steps+ 1)
        for _ in range(self.K_epochs):
            for t in range(self.n_steps-1):
                # ori_action = self.original_policy.step(old_states[:,t], time_steps[t].to(device), time_steps[t+1].to(device))
                # differnce = torch.norm(ori_action - old_actions[:,t+1])/100
                # print(differnce)
                # rewards -= differnce
                # Monte Carlo estimate of returns

                # print(rewards.shape, old_state_values[:,t].shape)
                
                # calculate advantages
                advantages = rewards.detach() - old_state_values[:,t].detach()
                
                # Evaluating old actions and values
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states[:,t], old_actions[:,t], time_steps[t].to(device), time_steps[t+1].to(device))

                # match state_values tensor dimensions with rewards tensor
                state_values = torch.squeeze(state_values)
                
                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - old_logprobs[:,t].detach())

                # Finding Surrogate Loss   
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

                # final loss of clipped objective PPO
                loss = -torch.min(surr1, surr2) + 0.01 * dist_entropy + self.MseLoss(state_values, rewards) 
                # print(surr1.mean().item(), surr2.mean().item(), self.MseLoss(state_values, rewards).item(), dist_entropy.mean().item())
                # take gradient step
                # print(loss.mean().item())
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
                # if self.start_actor:
                #     self.actor_optimizer.zero_grad()
                #     loss.mean().backward()
                #     self.actor_optimizer.step()


                # self.critic_optimizer.zero_grad()
                # value_loss = self.MseLoss(state_values, rewards)
                # value_loss.backward()

        # Copy new weights into old policy
        
        # clear buffer
        # self.buffer.clear()
    
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
        