import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data.sampler import *

def v_limited(Vraw, Vlim):
    r = np.sqrt(Vraw[:, 0] ** 2 + Vraw[:, 1] ** 2)
    mask = r <= Vlim
    Vlimit = np.where(mask[:, np.newaxis], Vraw, Vlim * Vraw / r[:, np.newaxis])
    return Vlimit


# 将二维数组每行元素相乘得到一维数组
def multiply_rows(arr):
    result = []
    for row in arr:
        product = 1
        for num in row:
            product *= num
        result.append(product)
    return np.array(result)


# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)

class Actor_RNN(nn.Module):
    def __init__(self, args, actor_input_dim):
        super(Actor_RNN, self).__init__()

        # 自动计算新维度
        self.actor_input_dim = args.obs_dim  # 从 args 读取新维度
        if self.add_agent_id:
            self.actor_input_dim += args.N  # 如果添加智能体 ID，增加维度

        self.rnn_hidden = None
        self.fc1 = nn.Linear(actor_input_dim, args.rnn_hidden_dim)  # Actor_RNN
        # self.fc1 = nn.Linear(actor_input_dim, args.rnn_hidden_dim)   # --原始的输入维度
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        # self.fc2 = nn.Linear(args.rnn_hidden_dim, args.action_dim) # 离散动作

        self.fc_mu = torch.nn.Linear(args.rnn_hidden_dim, args.action_dim)  # 连续动作
        # self.fc_std = torch.nn.Linear(args.rnn_hidden_dim, args.action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            # orthogonal_init(self.fc2, gain=0.01)  # 离散
            orthogonal_init(self.fc_mu)
            # orthogonal_init(self.fc_std)

    def forward(self, actor_input):
        # When 'choose_action': actor_input.shape=(N, actor_input_dim), prob.shape=(N, action_dim)
        # When 'train':         actor_input.shape=(mini_batch_size*N, actor_input_dim),prob.shape=(mini_batch_size*N, action_dim)
        x = self.activate_func(self.fc1(actor_input))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        # prob = torch.softmax(self.fc2(self.rnn_hidden), dim=-1)
        mu = 1.0 * torch.tanh(self.fc_mu(self.rnn_hidden))
        # std = F.softplus(self.fc_std(self.rnn_hidden))
        log_std = self.log_std.expand_as(mu)  # To make 'log_std' have the same dimension as 'mean'
        std = torch.exp(log_std)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0

        # return prob
        return mu, std


class Critic_RNN(nn.Module):
    def __init__(self, args, critic_input_dim):
        super(Critic_RNN, self).__init__()
        self.rnn_hidden = None

        self.fc1 = nn.Linear(critic_input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, 1)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2)

    def forward(self, critic_input):
        # When 'get_value': critic_input.shape=(N, critic_input_dim), value.shape=(N, 1)
        # When 'train':     critic_input.shape=(mini_batch_size*N, critic_input_dim), value.shape=(mini_batch_size*N, 1)
        x = self.activate_func(self.fc1(critic_input))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        value = self.fc2(self.rnn_hidden)
        return value


#
class Actor_MLP(nn.Module):
    def __init__(self, args, actor_input_dim):
        super(Actor_MLP, self).__init__()
        self.fc1 = nn.Linear(actor_input_dim, args.mlp_hidden_dim)
        # self.fc1 = nn.Linear(actor_input_dim, args.mlp_hidden_dim)   # 原始的输入维度
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        # self.fc3 = nn.Linear(args.mlp_hidden_dim, args.action_dim)
        self.fc_mu = torch.nn.Linear(args.mlp_hidden_dim, args.action_dim)  # 连续动作
        # self.fc_std = torch.nn.Linear(args.mlp_hidden_dim, args.action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc_mu, gain=0.01)
            # orthogonal_init(self.fc_std, gain=0.01)
            # orthogonal_init(self.fc3, gain=0.01)

    def forward(self, actor_input):
        # When 'choose_action': actor_input.shape=(N, actor_input_dim), prob.shape=(N, action_dim)
        # When 'train':         actor_input.shape=(mini_batch_size, episode_limit, N, actor_input_dim), prob.shape(mini_batch_size, episode_limit, N, action_dim)
        x = self.activate_func(self.fc1(actor_input))
        x = self.activate_func(self.fc2(x))
        mu = 2.0 * torch.tanh(self.fc_mu(x))
        log_std = self.log_std.expand_as(mu)  # To make 'log_std' have the same dimension as 'mean'
        std = torch.exp(log_std)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
        # std = F.softplus(self.fc_std(x))
        # prob = torch.softmax(self.fc3(x), dim=-1)
        # return prob
        return mu, std


class Critic_MLP(nn.Module):
    def __init__(self, args, critic_input_dim):
        super(Critic_MLP, self).__init__()
        mlp_hidden_dim = 256
        # self.fc1 = nn.Linear(critic_input_dim, args.mlp_hidden_dim)
        # self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        # self.fc3 = nn.Linear(args.mlp_hidden_dim, 1)
        # self.fc1 = nn.Linear(15, args.mlp_hidden_dim)  # Actor_MLP
        self.fc1 = nn.Linear(critic_input_dim, mlp_hidden_dim)  # 原始的输入维度
        self.fc2 = nn.Linear(mlp_hidden_dim, mlp_hidden_dim)
        self.fc3 = nn.Linear(mlp_hidden_dim, 1)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, critic_input):
        # When 'get_value': critic_input.shape=(N, critic_input_dim), value.shape=(N, 1)
        # When 'train':     critic_input.shape=(mini_batch_size, episode_limit, N, critic_input_dim), value.shape=(mini_batch_size, episode_limit, N, 1)
        x = self.activate_func(self.fc1(critic_input))
        x = self.activate_func(self.fc2(x))
        value = self.fc3(x)
        return value


class MAPPO_MPE:
    def __init__(self, args, device):
        self.N = args.N
        self.action_dim = args.action_dim
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.episode_limit = args.episode_limit
        self.rnn_hidden_dim = args.rnn_hidden_dim

        self.device = device

        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr = args.lr
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.epsilon = args.epsilon
        self.K_epochs = args.K_epochs
        self.entropy_coef = args.entropy_coef
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        self.use_rnn = args.use_rnn
        self.add_agent_id = args.add_agent_id
        self.use_value_clip = args.use_value_clip

        self.hidden_state = None
        self.critic_hidden_state = None

        self.max_action = 1.0

        # get the input dimension of actor and critic
        self.actor_input_dim = args.obs_dim
        self.critic_input_dim = args.state_dim
        if self.add_agent_id:
            print("------add agent id------")
            self.actor_input_dim += args.N
            self.critic_input_dim += args.N

        if self.use_rnn:
            print("------use rnn------")
            self.actor = Actor_RNN(args, self.actor_input_dim).to(self.device)
            self.critic = Critic_RNN(args, self.critic_input_dim).to(self.device)
        else:  # 默认是不使用RNN--->MLP
            self.actor = Actor_MLP(args, self.actor_input_dim).to(self.device)   # 基础的前馈神经网络
            self.critic = Critic_MLP(args, self.critic_input_dim).to(self.device)   # 基础的前馈神经网络
            self.ac_parameters = list(self.actor.parameters()) + list(self.critic.parameters())
            # self.actor = Actor_LSTM(actor_input_dim= self.action_dim, hidden_dim=args.rnn_hidden_dim, action_dim=args.action_dim, num_layers=1).to(args.device)

        if self.set_adam_eps:
            print("------set adam eps------")
            # self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.lr, eps=1e-5)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr, eps=1e-5)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr, eps=1e-5)
        else:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
            # self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.lr)

    # 返回每个智能体选取动作的序号及其对应的logp  --原始版本MLP
    def choose_action(self, obs_n, evaluate):
        with torch.no_grad():
            actor_inputs = []
            obs_n = torch.tensor(np.array(obs_n), dtype=torch.float32).to(self.device)  # obs_n.shape=(N，obs_dim)
            actor_inputs.append(obs_n)
            if self.add_agent_id:
                actor_inputs.append(torch.eye(self.N))
            actor_inputs = torch.cat([x for x in actor_inputs], dim=-1)  # actor_input.shape=(N, actor_input_dim)
            mu_n, std = self.actor(actor_inputs)
            action_dist = torch.distributions.Normal(mu_n, std)
            if evaluate:  # When evaluating the policy, we select the action with the highest probability
                a_n = action_dist.sample()
                return a_n, None
            else:
                a_n = action_dist.sample()
                a_logprob_n = action_dist.log_prob(a_n)
                a_logprob_n = np.sum(a_logprob_n.detach().cpu().numpy(), axis=-1)
                return a_n.detach().cpu(), a_logprob_n

    # 获取每个智能体当前状态的state value --原始版本MLP
    def get_value(self, s):
        with torch.no_grad():
            critic_inputs = []
            # Because each agent has the same global state, we need to repeat the global state 'N' times.
            s = torch.tensor(s, dtype=torch.float32).unsqueeze(0).repeat(self.N, 1)  # (state_dim,)-->(N,state_dim)
            critic_inputs.append(s)
            if self.add_agent_id:  # Add an one-hot vector to represent the agent_id
                critic_inputs.append(torch.eye(self.N))
            critic_inputs = torch.cat([x for x in critic_inputs], dim=-1).to(
                self.device)  # critic_input.shape=(N, critic_input_dim)
            v_n = self.critic(critic_inputs).cpu()  # v_n.shape(N,1)
            return v_n.numpy().flatten()

    # 模型训练
    def train(self, replay_buffer, total_steps):
        batch = replay_buffer.get_training_data()  # get training data
        # self.agent_n.reset_hidden_state()  # 未解析的引用

        # Calculate the advantage using GAE
        adv = []
        gae = 0
        with torch.no_grad():  # adv and td_target have no gradient
            deltas = batch['r_n'] + self.gamma * batch['v_n'][:, 1:] * (1 - batch['done_n']) - batch['v_n'][:,
                                                                                               :-1]  # deltas.shape=(batch_size,episode_limit,N)
            for t in reversed(range(self.episode_limit)):
                gae = deltas[:, t] + self.gamma * self.lamda * gae
                adv.insert(0, gae)
            adv = torch.stack(adv, dim=1)  # adv.shape(batch_size,episode_limit,N)
            v_target = adv + batch['v_n'][:, :-1]  # v_target.shape(batch_size,episode_limit,N)
            if self.use_adv_norm:  # Trick 1: advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        """
            Get actor_inputs and critic_inputs
            actor_inputs.shape=(batch_size, max_episode_len, N, actor_input_dim)
            critic_inputs.shape=(batch_size, max_episode_len, N, critic_input_dim)
        """
        actor_inputs, critic_inputs = self.get_inputs(batch)

        # Initialize lists to store losses for this training call
        actor_losses_epoch = []
        critic_losses_epoch = []

        # Optimize policy for K epochs:
        '''for _ in range(self.K_epochs):
            for index in BatchSampler(SequentialSampler(range(self.batch_size)), self.mini_batch_size, False):
                if self.use_rnn:
                    # If use RNN, we need to reset the rnn_hidden of the actor and critic.
                    self.actor.rnn_hidden = None
                    self.critic.rnn_hidden = None
                    actions_mean, actions_std, values_now = [], [], []
                    for t in range(self.episode_limit):
                        action_mean, action_std = self.actor(
                            actor_inputs[index, t].reshape(self.mini_batch_size * self.N,
                                                           -1))  # prob.shape=(mini_batch_size*N, action_dim)
                        actions_mean.append(action_mean.reshape(self.mini_batch_size, self.N,
                                                                -1))  # prob.shape=(mini_batch_size,N,action_dim）
                        actions_std.append(action_std.reshape(self.mini_batch_size, self.N, -1))
                        v = self.critic(critic_inputs[index, t].reshape(self.mini_batch_size * self.N,
                                                                        -1))  # v.shape=(mini_batch_size*N,1)
                        values_now.append(v.reshape(self.mini_batch_size, self.N))  # v.shape=(mini_batch_size,N)
                    # Stack them according to the time (dim=1)
                    actions_mean = torch.stack(actions_mean, dim=1)
                    actions_std = torch.stack(actions_std, dim=1)
                    values_now = torch.stack(values_now, dim=1)
                else:
                    actions_mean, actions_std = self.actor(actor_inputs[index])
                    values_now = self.critic(critic_inputs[index]).squeeze(-1)

                dist_now = Normal(actions_mean, actions_std)
                a_logprob_n_now = dist_now.log_prob(batch['a_n'][index])
                a_logprob_n_now = torch.sum(a_logprob_n_now, dim=-1)
                dist_entropy = dist_now.entropy()
                dist_entropy = torch.sum(dist_entropy, dim=-1).to(
                    self.device)  # dist_entropy.shape=(mini_batch_size, episode_limit, N)
                # batch['a_n'][index].shape=(mini_batch_size, episode_limit, N)
                # a/b=exp(log(a)-log(b))
                ratios = torch.exp(a_logprob_n_now - batch['a_logprob_n'][
                    index].detach())  # ratios.shape=(mini_batch_size, episode_limit, N)
                surr1 = ratios * adv[index]
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy
                self.actor_optimizer.zero_grad()
                actor_loss = actor_loss.mean()
                actor_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
                self.actor_optimizer.step()

                if self.use_value_clip:
                    values_old = batch["v_n"][index, :-1].detach()
                    values_error_clip = torch.clamp(values_now - values_old, -self.epsilon, self.epsilon) + values_old - \
                                        v_target[index]
                    values_error_original = values_now - v_target[index]
                    critic_loss = torch.max(values_error_clip ** 2, values_error_original ** 2)
                else:
                    critic_loss = (values_now - v_target[index]) ** 2
                self.critic_optimizer.zero_grad()
                critic_loss = critic_loss.mean()
                critic_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
                self.critic_optimizer.step()'''
        for _ in range(self.K_epochs):
            for index in BatchSampler(SequentialSampler(range(self.batch_size)), self.mini_batch_size, False):
                # ... (RNN specific logic if self.use_rnn else MLP logic) ...
                if self.use_rnn:
                    # ... (RNN forward pass) ...
                    pass  # Placeholder for your RNN logic
                else:
                    actions_mean, actions_std = self.actor(actor_inputs[index])
                    values_now = self.critic(critic_inputs[index]).squeeze(-1)

                dist_now = Normal(actions_mean, actions_std)
                a_logprob_n_now = dist_now.log_prob(batch['a_n'][index])
                a_logprob_n_now = torch.sum(a_logprob_n_now, dim=-1)
                dist_entropy = dist_now.entropy()
                dist_entropy = torch.sum(dist_entropy, dim=-1).to(self.device)

                ratios = torch.exp(a_logprob_n_now - batch['a_logprob_n'][index].detach())
                surr1 = ratios * adv[index]
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy

                self.actor_optimizer.zero_grad()
                current_actor_loss = actor_loss.mean()  # Get the mean loss for this mini-batch
                current_actor_loss.backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
                self.actor_optimizer.step()
                actor_losses_epoch.append(current_actor_loss.item())  # Store scalar value

                if self.use_value_clip:
                    values_old = batch["v_n"][index, :-1].detach()
                    values_error_clip = torch.clamp(values_now - values_old, -self.epsilon, self.epsilon) + values_old - \
                                        v_target[index]
                    values_error_original = values_now - v_target[index]
                    critic_loss = torch.max(values_error_clip ** 2, values_error_original ** 2)
                else:
                    critic_loss = (values_now - v_target[index]) ** 2

                self.critic_optimizer.zero_grad()
                current_critic_loss = critic_loss.mean()  # Get the mean loss for this mini-batch
                current_critic_loss.backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
                self.critic_optimizer.step()
                critic_losses_epoch.append(current_critic_loss.item())  # Store scalar value
        if self.use_lr_decay:
            self.lr_decay(total_steps)

        # Return the average loss over all K_epochs and mini-batches for this training call
        # avg_actor_loss = np.mean(actor_losses_epoch) if actor_losses_epoch else 0
        # avg_critic_loss = np.mean(critic_losses_epoch) if critic_losses_epoch else 0

        avg_actor_loss = np.mean(actor_losses_epoch) if actor_losses_epoch else 0
        avg_critic_loss = np.mean(critic_losses_epoch) if critic_losses_epoch else 0

        # dist_now = None
        # try:
        #     # Debugging before creating the Normal distribution
        #     if torch.isnan(actions_mean).any() or torch.isnan(actions_std).any():
        #         raise ValueError("NaN detected in actions_mean or actions_std!")
        #     if torch.isinf(actions_mean).any() or torch.isinf(actions_std).any():
        #         raise ValueError("Inf detected in actions_mean or actions_std!")
        #     if (actions_std <= 0).any():
        #         raise ValueError("Invalid values in actions_std (<= 0 detected)!")
        #
        #     # Ensure actions_std has valid values
        #     actions_std = torch.clamp(actions_std, min=1e-6)
        #
        #     # Create the Normal distribution
        #     dist_now = Normal(actions_mean, actions_std)
        # except ValueError as e:
        #     print("Error when creating Normal distribution:", e)
        #     print("actions_mean:", actions_mean)
        #     print("actions_std:", actions_std)
        #     raise e  # Optional: Rethrow the exception if you want to terminate

        return avg_actor_loss, avg_critic_loss  # <--- MODIFIED: Return losses

    def lr_decay(self, total_steps):  # Trick 6: learning rate Decay
        lr_now = self.lr * (1 - total_steps / self.max_train_steps)
        for p in self.actor_optimizer.param_groups:
            p['lr'] = lr_now
        for pa in self.critic_optimizer.param_groups:
            pa['lr'] = lr_now
        # for p in self.ac_optimizer.param_groups:
        #     p['lr'] = lr_now

    def get_inputs(self, batch):
        actor_inputs, critic_inputs = [], []
        actor_inputs.append(batch['obs_n'])
        critic_inputs.append(batch['s'].unsqueeze(2).repeat(1, 1, self.N, 1))
        if self.add_agent_id:
            # agent_id_one_hot.shape=(mini_batch_size, max_episode_len, N, N)
            agent_id_one_hot = torch.eye(self.N).unsqueeze(0).unsqueeze(0).repeat(self.batch_size, self.episode_limit,
                                                                                  1, 1)
            actor_inputs.append(agent_id_one_hot)
            critic_inputs.append(agent_id_one_hot)

        actor_inputs = torch.cat([x for x in actor_inputs], dim=-1).to(
            self.device)  # actor_inputs.shape=(batch_size, episode_limit, N, actor_input_dim)
        critic_inputs = torch.cat([x for x in critic_inputs], dim=-1).to(
            self.device)  # critic_inputs.shape=(batch_size, episode_limit, N, critic_input_dim)
        return actor_inputs, critic_inputs

    def save_model(self, env_name, number, seed, total_steps):
        torch.save(self.actor.state_dict(),
                   "./model/MAPPO_actor_env_{}_number_{}_seed_{}_step_{}k.pth".format(env_name, number, seed,
                                                                                      int(total_steps / 1000)))

    def load_model(self, env_name, number, seed, step):
        self.actor.load_state_dict(torch.load(
            "./model/MAPPO_actor_env_{}_number_{}_seed_{}_step_{}k.pth".format(env_name, number, seed, step)))
