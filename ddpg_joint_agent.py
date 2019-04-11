import torch
import os
from datetime import datetime
import numpy as np
from mpi4py import MPI
from models import actor, critic
from utils import sync_networks, sync_grads
from replay_buffer import replay_buffer
from normalizer import normalizer
from her import her_sampler

"""
ddpg with HER (MPI-version)

"""
def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
    o_norm = np.clip((o - o_mean) / (o_std), -args.clip_range, args.clip_range)
    g_norm = np.clip((g - g_mean) / (g_std), -args.clip_range, args.clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs

class ddpg_joint_agent:
    def __init__(self, args, envs_lst, env_params, expert_lst_dir):
        self.args = args
        self.envs_lst = envs_lst
        self.env_params = env_params

        # initialize expert
        self.expert_lst = []
        for dir in expert_lst_dir:
            expert_load_path = dir + '/model.pt'
            o_mean, o_std, g_mean, g_std, model = torch.load(expert_load_path)
            expert_model = actor(env_params, env_params['obs'] + env_params['goal'])
            expert_model.load_state_dict(model)
            self.expert_lst.append({"model": expert_model, "o_mean": o_mean, "o_std": o_std, "g_mean": g_mean, "g_std": g_std})

        # create the network
        self.actor_network = actor(env_params, env_params['obs'] + env_params['goal'] + env_params['action'])
        self.critic_network = critic(env_params, env_params['obs'] + env_params['goal'] + 2 * env_params['action'])

        # create the normalizer
        self.o_norm = normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range)
        self.sg_norm = normalizer(size=env_params['action'], default_clip_range=self.args.clip_range)

        # load model if load_path is not None
        if self.args.load_dir != '':
            load_path = self.args.load_dir + '/model.pt'
            o_mean, o_std, g_mean, g_std, sg_mean, sg_std, model = torch.load(load_path)
            self.o_norm.mean = o_mean
            self.o_norm.std = o_std
            self.g_norm.mean = g_mean
            self.g_norm.std = g_std
            self.sg_norm.mean = sg_mean
            self.sg_norm.std = sg_std
            self.actor_network.load_state_dict(model)

        # sync the networks across the cpus
        sync_networks(self.actor_network)
        sync_networks(self.critic_network)
        # build up the target network
        self.actor_target_network = actor(env_params, env_params['obs'] + env_params['goal'] + env_params['action'])
        self.critic_target_network = critic(env_params, env_params['obs'] + env_params['goal'] + 2 * env_params['action'])
        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # if use gpu
        if self.args.cuda:
            self.actor_network.cuda()
            self.critic_network.cuda()
            self.actor_target_network.cuda()
            self.critic_target_network.cuda()
        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)
        # her sampler
        self.her_module_lst = [her_sampler(self.args.replay_strategy, self.args.replay_k, env.compute_reward) for env in self.envs_lst]
        # create the replay buffer
        self.buffer_lst = [replay_buffer(self.env_params, self.args.buffer_size, her_module.sample_her_transitions) in self.her_module_lst]

        # path to save the model
        self.model_path = os.path.join(self.args.save_dir, self.args.env_name)

    def learn(self):
        """
        train the network

        """
        # start to collect samples
        for epoch in range(self.args.n_epochs):
            for env, expert, her_module, buffer in zip(self.envs_lst, self.expert_lst, self.her_module_lst, self.buffer_lst):
                for _ in range(self.args.n_cycles):
                    mb_obs, mb_ag, mb_g, mb_sg, mb_actions = [], [], [], [], []
                    for _ in range(self.args.num_rollouts_per_mpi):
                        # reset the rollouts
                        ep_obs, ep_ag, ep_g, ep_sg, ep_actions = [], [], [], [], []
                        # reset the environment
                        observation = env.reset()
                        obs = observation['observation']
                        ag = observation['achieved_goal']
                        g = observation['desired_goal']
                        with torch.no_grad():
                            input = process_inputs(obs, g, expert['o_mean'], expert['o_std'], expert['g_mean'], expert['g_std'], self.args)
                            expert_policy = expert["model"](input).cpu().numpy().squeeze()
                        # start to collect samples
                        for _ in range(self.env_params['max_timesteps']):
                            with torch.no_grad():
                                input_tensor = self._preproc_inputs(obs, g)
                                pi = self.actor_network(input_tensor)
                                action = self._select_actions(pi)
                            # feed the actions into the environment
                            observation_new, _, _, info = env.step(action)
                            obs_new = observation_new['observation']
                            ag_new = observation_new['achieved_goal']

                            with torch.no_grad():
                                input_new = process_inputs(obs_new, g, expert['o_mean'], expert['o_std'], expert['g_mean'], expert['g_std'], self.args)
                                expert_policy_new = expert["model"](input_new).cpu().numpy().squeeze()
                            # append rollouts
                            ep_obs.append(obs.copy())
                            ep_ag.append(ag.copy())
                            ep_g.append(g.copy())
                            ep_sg.append(expert_policy.copy())
                            ep_actions.append(action.copy())
                            # re-assign the observation
                            obs = obs_new
                            ag = ag_new
                            expert_policy = expert_policy_new
                        ep_obs.append(obs.copy())
                        ep_ag.append(ag.copy())
                        mb_obs.append(ep_obs)
                        mb_ag.append(ep_ag)
                        mb_g.append(ep_g)
                        mb_sg.append(ep_sg)
                        mb_actions.append(ep_actions)
                    # convert them into arrays
                    mb_obs = np.array(mb_obs)
                    mb_ag = np.array(mb_ag)
                    mb_g = np.array(mb_g)
                    mb_sg = np.array(mb_sg)
                    mb_actions = np.array(mb_actions)
                    # store the episodes
                    buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions, mb_sg])
                    self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions, mb_sg], her_module)
                    for _ in range(self.args.n_batches):
                        # train the network
                        self._update_network(buffer)
                    # soft update
                    self._soft_update_target_network(self.actor_target_network, self.actor_network)
                    self._soft_update_target_network(self.critic_target_network, self.critic_network)
                # start to do the evaluation
                success_rate = self._eval_agent()
                if MPI.COMM_WORLD.Get_rank() == 0:
                    print('[{}] epoch is: {}, eval success rate is: {:.3f}'.format(datetime.now(), epoch, success_rate))
                    torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, self.actor_network.state_dict()], \
                                self.model_path + '/model.pt')

    # pre_process the inputs
    def _preproc_inputs(self, obs, g, sg):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        sg_norm = self.sg_norm.normalize(sg)
        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, g_norm, sg_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
        return inputs

    # this function will choose action for the agent and do the exploration
    def _select_actions(self, pi):
        action = pi.cpu().numpy().squeeze()
        # add the gaussian
        action += self.args.noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])
        # random actions...
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                            size=self.env_params['action'])
        # choose if use the random actions
        action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_actions - action)
        return action

    # update the normalizer
    def _update_normalizer(self, episode_batch, her_module):
        mb_obs, mb_ag, mb_g, mb_actions, mb_sg = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        mb_sg_next = mb_sg[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs,
                       'ag': mb_ag,
                       'g': mb_g,
                       'actions': mb_actions,
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       'sg': mb_sg,
                       'sg_next': mb_sg_next,
                       }
        transitions = her_module.sample_her_transitions(buffer_temp, num_transitions)
        obs, g, sg = transitions['obs'], transitions['g'], transitions['sg']
        # pre process the obs and g
        transitions['obs'], transitions['g'], transitions['sg'] = self._preproc_og(obs, g, sg)
        # update
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        self.sg_norm.update(transitions['sg'])
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()
        self.sg_norm.recompute_stats()

    def _preproc_og(self, o, g, sg):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        sg = np.clip(sg, -self.args.clip_obs, self.args.clip_obs)
        return o, g, sg

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    # update the network
    def _update_network(self, buffer):
        # sample the episodes
        transitions = buffer.sample(self.args.batch_size)
        # pre-process the observation and goal
        o, o_next, g, sg, sg_next = transitions['obs'], transitions['obs_next'], transitions['g'], transitions['sg'], transitions['sg_next']
        transitions['obs'], transitions['g'], transitions['sg'] = self._preproc_og(o, g, sg)
        transitions['obs_next'], transitions['g_next'], transitions['sg_next'] = self._preproc_og(o_next, g, sg_next)
        # start to do the update
        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        sg_norm = self.sg_norm.normalize(transitions['sg'])
        inputs_norm = np.concatenate([obs_norm, g_norm, sg_norm], axis=1)
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        sg_next_norm = self.sg_norm.normalize(transitions['sg_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm, sg_next_norm], axis=1)
        # transfer them into the tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32)
        if self.args.cuda:
            inputs_norm_tensor = inputs_norm_tensor.cuda()
            inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()
        # calculate the target Q value function
        with torch.no_grad():
            # do the normalization
            # concatenate the stuffs
            actions_next = self.actor_target_network(inputs_next_norm_tensor)
            q_next_value = self.critic_target_network(inputs_next_norm_tensor, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + self.args.gamma * q_next_value
            target_q_value = target_q_value.detach()
            # clip the q value
            clip_return = 1 / (1 - self.args.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)
        # the q loss
        real_q_value = self.critic_network(inputs_norm_tensor, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()
        # the actor loss
        actions_real = self.actor_network(inputs_norm_tensor)
        actor_loss = -self.critic_network(inputs_norm_tensor, actions_real).mean()
        actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()
        # start to update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor_network)
        self.actor_optim.step()
        # update the critic_network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic_network)
        self.critic_optim.step()

    # do the evaluation
    def _eval_agent(self):
        total_success_rate = []
        for env, expert in zip(self.envs_lst, self.expert_lst):
            for _ in range(self.args.n_test_rollouts):
                per_success_rate = []
                observation = env.reset()
                obs = observation['observation']
                g = observation['desired_goal']
                with torch.no_grad():
                    input = process_inputs(obs, g, expert['o_mean'], expert['o_std'], expert['g_mean'], expert['g_std'], self.args)
                    sg = expert["model"](input).cpu().numpy().squeeze()
                for _ in range(self.env_params['max_timesteps']):
                    with torch.no_grad():
                        input_tensor = self._preproc_inputs(obs, g, sg)
                        pi = self.actor_network(input_tensor)
                        # convert the actions
                        actions = pi.detach().cpu().numpy().squeeze()
                    observation_new, _, _, info = env.step(actions)
                    obs = observation_new['observation']
                    g = observation_new['desired_goal']
                    with torch.no_grad():
                        input = process_inputs(obs, g, expert['o_mean'], expert['o_std'], expert['g_mean'], expert['g_std'], self.args)
                        sg = expert["model"](input).cpu().numpy().squeeze()
                    per_success_rate.append(info['is_success'])
                total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size()
