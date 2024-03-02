import numpy as np
import copy
import torch
from torch.optim import Adam
from torch.nn.functional import relu
import os 
from time import time

def weights_init_(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.1)


def numpy_to_torch(numpy_input):
    if isinstance(numpy_input, tuple):
        return tuple(numpy_to_torch(e) for e in numpy_input)
    return torch.from_numpy(numpy_input).float()


def torch_to_numpy(torch_input):
    if isinstance(torch_input, tuple):
        return tuple(torch_to_numpy(e) for e in torch_input)
    return torch_input.data.numpy()


class MLP(torch.nn.Module):
    def __init__(self, hidden_sizes, output_size, input_size):
        super().__init__()

        self.hidden_activation = relu
        self.output_activation = lambda x: x
        self.fcs = torch.nn.ModuleList()
        self.out_size = output_size
        self.in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = torch.nn.Linear(self.in_size, next_size)
            self.in_size = next_size
            self.fcs.append(fc)
        self.last_fc = torch.nn.Linear(self.in_size, self.out_size)

    def forward(self, net_input):
        h = net_input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            h = self.hidden_activation(h)
        logits = self.last_fc(h)
        output = self.output_activation(logits)
        return output


class QNetwork(MLP):
    def __init__(self, hidden_sizes, output_size, input_size):
        super().__init__(hidden_sizes, output_size, input_size)

        self.apply(weights_init_)

    def forward(self, state, action):
        flat_inputs = torch.cat((state, action), dim=1)
        return super().forward(flat_inputs)


class ValueNetwork(MLP):
    def __init__(self, hidden_sizes, output_size, input_size):
        super().__init__(hidden_sizes, output_size, input_size)

        self.apply(weights_init_)

    def forward(self, state):
        return super().forward(state)


class PolicyNetwork(MLP):
    def __init__(self, hidden_sizes, output_size, input_size):
        super().__init__(hidden_sizes, output_size, input_size)
        self.last_fc_mean = self.last_fc
        self.last_fc_log_std = torch.nn.Linear(self.in_size, self.out_size)
        self.epsilon = 1e-6
        self.apply(weights_init_)

    def get_action(self, state, deterministic):
        action = self.forward(state.unsqueeze(0),
                              deterministic=deterministic)[0]
        return action.detach()

    def forward(self, state, deterministic=False):
        h = state
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            h = self.hidden_activation(h)

        if deterministic:
            mean = self.last_fc_mean(h)
            log_std = None
            log_prob = None
            action = torch.tanh(mean)

        else:
            mean = self.last_fc_mean(h)
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, -20, 2)
            std = torch.exp(log_std)
            normal = torch.distributions.Normal(mean, std)
            reparameterized_sample = normal.rsample()
            action = torch.tanh(reparameterized_sample)
            log_prob = normal.log_prob(reparameterized_sample)
            log_prob -= torch.log(1 - action.pow(2) + self.epsilon)
            log_prob = log_prob.sum(1, keepdim=True)

        return action, mean, log_std, log_prob


class InitPolicy:
    def __init__(self):
        self.function = torch.distributions.Uniform(0, 1)

    def get_action(self, *args, **kwargs):
        return self.function.sample()


class ReplayBuffer:
    def __init__(self, buffer_capacity, batch_size, state_dim, action_dim):
        """
        Store and sample experience using a replay buffer.
        Advantages: More efficient use of previous experience, by learning
                    with it multiple times. Better convergence behaviour
                    when training a function approximator.
        :param buffer_capacity: max. capacity of the replay buffer/
                                max. number of experience tuples (int)
        :param state_dim: Dimension of state (int)
        :param action_dim: Dimension of action (int)
        """
        self.capacity = buffer_capacity
        self.current_size = 0
        self.batch_size = batch_size
        self.index = 0

        self.S = np.zeros((buffer_capacity, state_dim), dtype=np.float32)
        self.A = np.zeros((buffer_capacity, action_dim), dtype=np.float32)
        self.R = np.zeros((buffer_capacity, 1), dtype=np.float32)
        self.S_prime = np.zeros((buffer_capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((buffer_capacity, 1), dtype=np.float32)

    def add_experience(self, experience):
        """
        Store experience in replay buffer.
        :param experience: tuple of (State, Action, Reward, next State)
        """
        self.index = (self.index + 1) % self.capacity
        self.current_size = min((self.current_size + 1), self.capacity)

        s, a, r, s_prime, done = experience
        self.S[self.index] = s
        self.A[self.index] = a
        self.R[self.index] = r
        self.S_prime[self.index] = s_prime
        self.done[self.index] = done

    def sample_minibatch(self):
        """
        Randomly sample minibatch of experience from replay buffer.
        :param batch_size: number of experiences to sample in a minibatch
        :return: tuple of (States(batch_size, state_dim),
                           Actions(batch_size, action_dim),
                           Rewards(batch_size, 1),
                           next States(batch_size, state_dim))
        """
        indices = np.random.choice(self.current_size, self.batch_size)
        S = self.S[indices]
        A = self.A[indices]
        R = self.R[indices]
        S_prime = self.S_prime[indices]
        done = self.done[indices]
        return S, A, R, S_prime, done


class SAC:
    def __init__(self, environment, policy_function, q1_function, q2_function,
                 value_function, replay_buffer, adam_learning_rate,
                 target_entropy, discount_factor_gamma,
                 soft_update_factor_tau):

        self.env = environment
        self.policy_function = policy_function
        self.q1_function = q1_function
        self.q2_function = q2_function
        self.value_function = value_function
        self.target_value_function = copy.deepcopy(self.value_function)
        self.log_alpha = torch.zeros(1, requires_grad=True)

        self.policy_function_optimizer = Adam(
            self.policy_function.parameters(), lr=adam_learning_rate)
        self.q1_function_optimizer = Adam(self.q1_function.parameters(),
                                          lr=adam_learning_rate)
        self.q2_function_optimizer = Adam(self.q2_function.parameters(),
                                          lr=adam_learning_rate)
        self.value_function_optimizer = Adam(self.value_function.parameters(),
                                             lr=adam_learning_rate)
        self.alpha_optimizer = Adam([self.log_alpha], lr=adam_learning_rate)

        self.replay_buffer = replay_buffer
        self.target_entropy = target_entropy
        self.gamma = discount_factor_gamma
        self.tau = soft_update_factor_tau

        self.loss_criterion = torch.nn.MSELoss()
        self.losses = {
            'q1_loss': 0,
            'q2_loss': 0,
            'value_loss': 0,
            'policy_loss': 0,
            'alpha_loss': 0
        }

    def do_training(self, num_steps_per_epoch):
        state = numpy_to_torch(self.env.reset())
        for i in range(num_steps_per_epoch):
            action = self.policy_function.get_action(state,
                                                     deterministic=False)
            action = torch_to_numpy(action).reshape(
                self.env.action_space.shape)
            next_state, reward, done, _ = self.env.step(action)
            experience = (state, action, reward, next_state, done)
            self.replay_buffer.add_experience(experience)

            batch = self.replay_buffer.sample_minibatch()
            batch = numpy_to_torch(batch)
            self.update_parameters_for_batch(batch)
            # perhaps update value function not every epoch
            self.update_value_function()

            state = numpy_to_torch(next_state)
            if done:
                state = numpy_to_torch(self.env.reset())

    def do_evaluation(self,
                      num_evaluation_episodes,
                      record=False,
                      save_dname=None):
        random_state_np = self.env.np_random.get_state()

        if record:
            self.do_visualization(record, save_dname)

        self.env.seed(2)

        returns = []
        potential = []

        forward = []
        energy = []
        jerk = []
        safety = []

        dones = 0
        state = numpy_to_torch(self.env.reset())
        for _ in range(num_evaluation_episodes):
            episode_return = 0

            potential_curr_state = 0 #value from boundaryDist function
            reward_forward = 0
            reward_energy = 0
            reward_jerk = 0
            reward_safety = 0  

            while True:
                action = self.policy_function.get_action(state,
                                                         deterministic=True)
                action = torch_to_numpy(action).reshape(
                    self.env.action_space.shape)
                next_state, reward, done, info = self.env.step(action)
                episode_return += reward
                potential_curr_state += (info['potential_curr_state'].val)

                reward_forward += info['reward_list'][0]
                reward_energy += info['reward_list'][1]
                reward_jerk += info['reward_list'][2]
                reward_safety += info['reward_list'][3]


                state = numpy_to_torch(next_state)

                if done:
                    state = numpy_to_torch(self.env.reset())
                    returns.append(episode_return)

                    potential.append(potential_curr_state)
                    forward.append(reward_forward)
                    energy.append(reward_energy)
                    jerk.append(reward_jerk)
                    safety.append(reward_safety)

                    dones += 1
                    break

        self.env.np_random.set_state(random_state_np)

        return np.mean(returns), dones, np.mean(potential), np.mean(forward), np.mean(energy), np.mean(jerk), np.mean(safety)

    def do_visualization(self, record=False, save_dname=None):
        if record:
            from gym import wrappers
            from time import time
            import os

            time_name = str(time())
            save_dname_vid = os.path.join(save_dname,
                                      'videos/' + time_name + '/')
            #env = wrappers.Monitor(self.env, save_dname)
            env = wrappers.RecordVideo(self.env, save_dname_vid)
        else:
            env = self.env
        env.seed(2)

        state = numpy_to_torch(env.reset())
        episode_return = 0 #accumulated reward 
        steps = 0 
        number_violations = 0

        history = {'accumulated_reward' : [],
                   'reward_per_step': [],
                   'timesteps': [],
                   'number_violations': 0,
                   'dist_from_speed_limit':[], 
                   'position' : [], 
                   'potential' : [],
                   'potential_id' : []}

        
        path_time = os.path.join(save_dname, 'evals/'+ time_name)
        os.makedirs(path_time, exist_ok=True)


        print(f'{"timestep":>11}|{"acc":>11}|{"prev acc":>11}| {"next speed limit":>11}|{"next speed limit pos":>11}|{"curr speed limit":>11}|{"curr speed limit pos":>11}|{"prev speed limit":>11}|{"prev speed limit pos":>11}|{"track length":>11}|{"velocity":>11}|{"position":>11}| {"safety ID":>11}|{"safety value":>11}'  
            
            + "\n" + 59 * "_",
            file=open(os.path.join(path_time, f'evalnum_curr.out'), 'w'))
        

        while True:
            action = self.policy_function.get_action(state, deterministic=True)
            action = torch_to_numpy(action).reshape(env.action_space.shape)
            next_state, reward, done, info = env.step(action)
            episode_return += reward
            steps += 1

            dist_from_s = env.current_speed_limit - env.velocity 
            #in m/s 
            

            state = numpy_to_torch(next_state)
            env.render()
            number_violations += env.num_violations

            history['accumulated_reward'].append(episode_return)
            history['reward_per_step'].append(reward)
            history['timesteps'].append(steps)
            history['number_violations'] = number_violations
            #violations are only if the km/h value is greater (used math.floor to get to integer value)
            history['dist_from_speed_limit'].append(dist_from_s)
            history['position'].append(info['position'])
            history['potential'].append(info['potential_curr_state'].val)
            history['potential_id'].append(info['potential_curr_state'].id)

            print(f'{steps:>11}',
                f'{env.monitor_curr["acceleration"]:>11}',
                f'{env.monitor_curr["prev_acceleration"]:>11}',
                f'{env.monitor_curr["next_speed_limit"]:>11}',
                f'{env.monitor_curr["next_speed_limit_position"]:>11}',
                f'{env.monitor_curr["current_speed_limit"]:>11}',
                f'{env.monitor_curr["current_speed_limit_position"]:>11}',
                f'{env.monitor_curr["previous_speed_limit"]:>11}',
                f'{env.monitor_curr["previous_speed_limit_position"]:>11}',
                f'{env.monitor_curr["track_length"]:>11}',
                f'{env.monitor_curr["velocity"]:>11}',
                f'{env.monitor_curr["position"]:>11}',
                f'{info["potential_curr_state"].id:>11}',
                f'{info["potential_curr_state"].val:>11}',
                file=open(os.path.join(path_time, f'evalnum_curr.out'), 'a'))
            

            np.save(os.path.join(save_dname_vid, f'eval.npy'), history)

            if done:
                state = numpy_to_torch(env.reset())
                break
        env.close()

    def init_replay_buffer(self, init_policy, fill_size):
        state = numpy_to_torch(self.env.reset())
        for _ in range(int(fill_size)):
            action = init_policy.get_action(state, deterministic=False)
            action = torch_to_numpy(action).reshape(
                self.env.action_space.shape)
            next_state, reward, done, _ = self.env.step(action)
            experience = (state, action, reward, next_state, done)
            self.replay_buffer.add_experience(experience)

            state = numpy_to_torch(next_state)
            if done:
                state = numpy_to_torch(self.env.reset())

    def update_parameters_for_batch(self, batch):
        (state, action, reward, next_state, done) = batch

        q1_pred = self.q1_function(state, action)
        q2_pred = self.q2_function(state, action)
        value_pred = self.value_function(state)

        (new_action, policy_mean, policy_log_std,
         log_pi) = self.policy_function(state)

        # Alpha loss & update
        alpha_loss = -(self.log_alpha *
                       (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        alpha = self.log_alpha.exp()

        # Q-function loss
        next_value_pred = self.target_value_function(next_state)
        q_target = reward + (1 - done) * self.gamma * next_value_pred
        q1_function_loss = self.loss_criterion(q1_pred, q_target.detach())
        q2_function_loss = self.loss_criterion(q2_pred, q_target.detach())

        # Value-function loss
        q1_pred_new_action = self.q1_function(state, new_action)
        q2_pred_new_action = self.q2_function(state, new_action)
        q_pred_new_action = torch.min(q1_pred_new_action, q2_pred_new_action)
        value_target = q_pred_new_action - alpha * log_pi
        value_function_loss = self.loss_criterion(value_pred,
                                                  value_target.detach())

        # Policy loss
        #policy_function_loss = (alpha * log_pi - q_pred_new_action).mean()
        temp = alpha * log_pi - q_pred_new_action
        policy_function_loss = torch.mean(temp)

        # Perhaps add regularization loss
        # mean_reg_loss = 0.001 * policy_mean.pow(2).mean()
        # std_reg_loss = 0.001 * policy_log_std.pow(2).mean()
        # policy_function_loss += mean_reg_loss + mean_std_loss

        # Q-function, Value-function Policy-function update
        self.q1_function_optimizer.zero_grad()
        q1_function_loss.backward()
        #self.q1_function_optimizer.step()

        self.q2_function_optimizer.zero_grad()
        q2_function_loss.backward()
        #self.q2_function_optimizer.step()

        self.value_function_optimizer.zero_grad()
        value_function_loss.backward()
        #self.value_function_optimizer.step()

        self.policy_function_optimizer.zero_grad()
        policy_function_loss.backward()
        #self.policy_function_optimizer.step()

        self.q1_function_optimizer.step()
        self.q2_function_optimizer.step()
        self.value_function_optimizer.step()
        self.policy_function_optimizer.step()

        # save losses for history
        losses = [
            q1_function_loss, q2_function_loss, value_function_loss,
            policy_function_loss, alpha_loss
        ]
        for key, loss in zip(self.losses.keys(), losses):
            self.losses[key] = loss.data

    def update_value_function(self):
        for target_p, p in zip(self.target_value_function.parameters(),
                               self.value_function.parameters()):
            target_p.data.copy_(target_p.data * (1 - self.tau) +
                                p.data * self.tau)
