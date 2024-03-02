import numpy as np
import torch
import time
import sys
import os

import gym
import arguments
import sac

from torch.utils.tensorboard import SummaryWriter

#This file is used for EVALUATION 


torch.autograd.set_detect_anomaly(True)

sys.stderr.write("\x1b[2J\x1b[H")
print("Running script \n \n")
args = arguments.get_args()



if args.env_id == 'DeterministicTrack-v0':
    env = gym.make('gym_longicontrol:' + args.env_id,
                   car_id=args.car_id,
                   speed_limit_positions=args.speed_limit_positions,
                   speed_limits=args.speed_limits,
                   reward_weights=args.reward_weights,
                   energy_factor=args.energy_factor)

elif args.env_id == 'StochasticTrack-v0':
    env = gym.make('gym_longicontrol:' + args.env_id,
                   car_id=args.car_id,
                   reward_weights=args.reward_weights,
                   energy_factor=args.energy_factor)
elif args.env_id in ['PotentialTrack-v0', 'LogicalConstraint-v0', 'LogicalConstraintScaling-v0']: 
    env = gym.make('gym_longicontrol:' + args.env_id,
                   car_id=args.car_id,
                   speed_limit_positions=args.speed_limit_positions,
                   speed_limits=args.speed_limits,
                   reward_weights=args.reward_weights,
                   energy_factor=args.energy_factor)
    ##added these for the different reward functions 
else:
    raise NotImplementedError



'''
if args.env_id == 'DeterministicTrack-v0':
    env = gym.make('gym_longicontrol:' + args.env_id,
                   car_id=args.car_id,
                   speed_limit_positions=args.speed_limit_positions,
                   speed_limits=args.speed_limits,
                   reward_weights=args.reward_weights,
                   energy_factor=args.energy_factor)
elif args.env_id == 'StochasticTrack-v0':
    env = gym.make('gym_longicontrol:' + args.env_id,
                   car_id=args.car_id,
                   reward_weights=args.reward_weights,
                   energy_factor=args.energy_factor)
else:
    raise NotImplementedError
    
'''

env.seed(args.seed)
torch.manual_seed(args.seed)



state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
adam_lr = args.adam_lr
replay_buffer_capacity = int(args.replay_buffer_capacity)
buffer_init_part = 0.1
optimization_batch = int(args.optimization_batch)
discount_factor_gamma = args.discount_factor_gamma
soft_update_factor_tau = args.soft_update_factor_tau
num_steps_per_epoch = int(args.num_steps_per_epoch)
num_epochs = int(args.num_epochs)
num_evaluation_episodes = 10
target_entropy = -action_dim
hidden_network_sizes = args.hidden_layer_sizes
visualize = args.visualize
record_mode = args.record

replay_buffer = sac.ReplayBuffer(buffer_capacity=replay_buffer_capacity,
                                 batch_size=optimization_batch,
                                 state_dim=state_dim,
                                 action_dim=action_dim)

q1_network = sac.QNetwork(hidden_sizes=hidden_network_sizes,
                          output_size=1,
                          input_size=state_dim + action_dim)

q2_network = sac.QNetwork(hidden_sizes=hidden_network_sizes,
                          output_size=1,
                          input_size=state_dim + action_dim)

value_network = sac.ValueNetwork(hidden_sizes=hidden_network_sizes,
                                 output_size=1,
                                 input_size=state_dim)

policy_network = sac.PolicyNetwork(hidden_sizes=hidden_network_sizes,
                                   output_size=action_dim,
                                   input_size=state_dim)

agent = sac.SAC(environment=env,
                policy_function=policy_network,
                q1_function=q1_network,
                q2_function=q2_network,
                value_function=value_network,
                replay_buffer=replay_buffer,
                adam_learning_rate=adam_lr,
                target_entropy=target_entropy,
                discount_factor_gamma=discount_factor_gamma,
                soft_update_factor_tau=soft_update_factor_tau)

print("\n \n Finish initializing SAC models \n \n")
print("\n \n epochs: {} \n \n".format(num_epochs))

if args.load_id is None:

    print("\n \n If load_id is None \n \n")

    save_dname = os.path.join(os.path.dirname(__file__),
                              f'out/{args.env_id}/SAC_id{args.save_id}')
    if not os.path.exists(save_dname):
        os.makedirs(save_dname)

    print("\n \n Finish creating directory \n \n ")

    init_policy = sac.InitPolicy()

    print("\n \n Finish initializing policy \n \n")

    agent.init_replay_buffer(init_policy,
                             replay_buffer_capacity * buffer_init_part)

    print("\n \n Finish replay buffer \n \n")

    history = {
        'training_steps': [],
        'steps_per_s': [],
        'eval_return': [],
        'potential_curr_state' : [],
        'reward_forward' : [],
        'reward_energy': [],
        'reward_jerk' : [],
        'reward_safety' : [],
        'losses': {
            'q1_loss': [],
            'q2_loss': [],
            'value_loss': [],
            'policy_loss': [],
            'alpha_loss': []
        }
    }

    prev_epoch = 0
    print(f'{"epoch":>11}|{"steps":>11}|{"steps/s":>11}|' +
          f'{"return":>11}|{"value loss":>11}' + '\n' + 59 * '_',
          file=open(os.path.join(save_dname, f'seed{args.seed}.out'), 'w'))

else:
    print("\n \n Loading in state dicts \n \n ")

    load_dname = os.path.join(os.path.dirname(__file__),
                              f'out\\{args.env_id}\\SAC_id{args.load_id}')
    if not os.path.exists(load_dname):
        raise ValueError(
            f'A model for {args.env_id} with load_id={args.load_id} '
            'does not exist!')
    save_dname = load_dname
    checkpoint = torch.load(os.path.join(load_dname, f'seed{args.seed}.tar'), map_location=torch.device('cpu'))

    print("\n \n Finish loading in tar file \n \n")

    agent.q1_function.load_state_dict(checkpoint['q1_function_state_dict'])
    agent.q2_function.load_state_dict(checkpoint['q2_function_state_dict'])
    agent.value_function.load_state_dict(
        checkpoint['value_function_state_dict'])
    agent.policy_function.load_state_dict(
        checkpoint['policy_function_state_dict'])
    agent.log_alpha = checkpoint['log_alpha']
    agent.q1_function_optimizer.load_state_dict(
        checkpoint['q1_optimizer_state_dict'])
    agent.q2_function_optimizer.load_state_dict(
        checkpoint['q2_optimizer_state_dict'])
    agent.value_function_optimizer.load_state_dict(
        checkpoint['value_optimizer_state_dict'])
    agent.policy_function_optimizer.load_state_dict(
        checkpoint['policy_optimizer_state_dict'])
    agent.alpha_optimizer.load_state_dict(
        checkpoint['alpha_optimizer_state_dict'])

    print("\n \n Finish loading state dicts \n \n")

    if visualize:
        agent.do_visualization(record_mode, save_dname)
        exit()

    init_policy = agent.policy_function
    agent.init_replay_buffer(init_policy,
                             replay_buffer_capacity * buffer_init_part)

    history = checkpoint['history']

    prev_epoch = checkpoint['epoch']
    print(f'{"epoch":>11}|{"steps":>11}|{"steps/s":>11}|' +
          f'{"return":>11}|{"value loss":>11}' + '\n' + 59 * '_',
          file=open(os.path.join(save_dname, f'seed{args.seed}.out'), 'a'))

#setting up tensorboard
writer = SummaryWriter(f'runs/{args.env_id}-{args.save_id}-{args.seed}')
#changed to include which environment this is in using args.env_id

best_reward = -20000 

for epoch in range(num_epochs):
    print("\n \n Training epoch number {} \n \n".format(epoch))
    
    t0 = time.time()
    agent.do_training(num_steps_per_epoch)
    t1 = time.time()

    if epoch % 10 == 0:
        #runs a test episode (10 times)
        mean_return, dones, potential, reward_forward, reward_energy, reward_jerk, reward_safety = agent.do_evaluation(num_evaluation_episodes,
                                                                                                record_mode, save_dname)

        # history
        history['training_steps'].append(
            (prev_epoch + epoch + 1) * num_steps_per_epoch)
        history['steps_per_s'].append(int(num_steps_per_epoch / (t1 - t0)))
        history['eval_return'].append(mean_return)

        history['potential_curr_state'].append(potential)
        history['reward_forward'].append(reward_forward)
        history['reward_energy'].append(reward_energy)
        history['reward_jerk'].append(reward_jerk)
        history['reward_safety'].append(reward_safety)

        for key in history['losses']:
            history['losses'][key].append(agent.losses[key])
        np.save(os.path.join(save_dname, f'seed{args.seed}.npy'), history)

        if mean_return > best_reward: 
            best_reward = mean_return
            torch.save(
            {
                'q1_function_state_dict':
                agent.q1_function.state_dict(),
                'q2_function_state_dict':
                agent.q2_function.state_dict(),
                'value_function_state_dict':
                agent.value_function.state_dict(),
                'policy_function_state_dict':
                agent.policy_function.state_dict(),
                'log_alpha':
                agent.log_alpha,
                'q1_optimizer_state_dict':
                agent.q1_function_optimizer.state_dict(),
                'q2_optimizer_state_dict':
                agent.q2_function_optimizer.state_dict(),
                'value_optimizer_state_dict':
                agent.value_function_optimizer.state_dict(),
                'policy_optimizer_state_dict':
                agent.policy_function_optimizer.state_dict(),
                'alpha_optimizer_state_dict':
                agent.alpha_optimizer.state_dict(),
                'epoch':
                prev_epoch + epoch,
                'history':
                history
            }, os.path.join(save_dname, f'best_reward_model.tar'))

        # save agent
        torch.save(
            {
                'q1_function_state_dict':
                agent.q1_function.state_dict(),
                'q2_function_state_dict':
                agent.q2_function.state_dict(),
                'value_function_state_dict':
                agent.value_function.state_dict(),
                'policy_function_state_dict':
                agent.policy_function.state_dict(),
                'log_alpha':
                agent.log_alpha,
                'q1_optimizer_state_dict':
                agent.q1_function_optimizer.state_dict(),
                'q2_optimizer_state_dict':
                agent.q2_function_optimizer.state_dict(),
                'value_optimizer_state_dict':
                agent.value_function_optimizer.state_dict(),
                'policy_optimizer_state_dict':
                agent.policy_function_optimizer.state_dict(),
                'alpha_optimizer_state_dict':
                agent.alpha_optimizer.state_dict(),
                'epoch':
                prev_epoch + epoch,
                'history':
                history
            }, os.path.join(save_dname, f'seed{args.seed}.tar'))

        # terminal output
        print(f'{epoch:>11}',
              f'{history["training_steps"][-1]:>11}',
              f'{history["steps_per_s"][-1]:>11}',
              f'{history["eval_return"][-1]:>11.2f}',
              f'{history["losses"]["value_loss"][-1]:>11.5f}',
              file=open(os.path.join(save_dname, f'seed{args.seed}.out'), 'a'))

        writer.add_scalar('eval_return', mean_return, epoch)
        writer.add_scalar('potential_curr_state', potential, epoch) #average value of computing boundaryDist function of current state
        writer.add_scalar('reward_forward', reward_forward, epoch)
        writer.add_scalar('reward_energy', reward_energy, epoch)
        writer.add_scalar('reward_jerk', reward_jerk, epoch)
        writer.add_scalar('reward_safety', reward_safety, epoch) 
        #average reward from extra reward for safety from potential based shaping 
        #potential is calculated through boundaryDist, so safety = relationship btwn potential of current and previous state
        
