import torch
import os
from datetime import datetime
import numpy as np
from models import actor, actor_image, critic, critic_image
# from utils import sync_networks, sync_grads
from replay_buffer import replay_buffer
from normalizer import normalizer
from her import her_sampler

import time

"""
ddpg with HER (MPI-version)

"""

def next_q_estimator(input_tensor, box_pose_tensor):
    # counter = 1-input_tensor[:, -1:]
    gripper_state_tensor = input_tensor[:, :3]

    # next q = 1 if satisfied all conditions: x, y, z bound and norm magnitude bound.
    offset_tensor = box_pose_tensor - gripper_state_tensor
    # print(offset_tensor, box_pose_tensor, gripper_state_tensor)
    # check upper-lower bound for each of x, y, z
    x_offset = offset_tensor[:, 0]
    _below_x_upper = (x_offset <= 0.04)
    _beyond_x_lower = (x_offset >= -0.04)
    y_offset = offset_tensor[:, 1]
    _below_y_upper = (y_offset <= 0.045)
    _beyond_y_lower = (y_offset >= -0.045)
    z_offset = offset_tensor[:, 2]
    _below_z_upper = (z_offset <= 0.)
    _beyond_z_lower = (z_offset >= -0.086)

    # check norm magnitude with in range:
    norm = torch.norm(offset_tensor, dim=-1)
    _magnitude_in_range = (norm <= 0.087)

    next_q = torch.unsqueeze(_below_x_upper & _beyond_x_lower & \
            _below_y_upper & _beyond_y_lower & \
            _below_z_upper & _beyond_z_lower & \
            _magnitude_in_range, 1).float()
    return next_q

class actor_agent:
    def __init__(self, args, env, env_params, image=True):
        self.args = args
        self.env = env
        self.env_params = env_params
        self.image = image

        # create the network
        if self.image:
            self.actor_network = actor_image(env_params, env_params['obs'])
            self.critic_network = critic_image(env_params, env_params['obs'] + env_params['action'])
        else:
            self.actor_network = actor(env_params, env_params['obs'])
            self.critic_network = critic(env_params, env_params['obs'] + env_params['action'])

        # create the normalizer
        self.o_norm = normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)

        # load model if load_path is not None
        if self.args.load_dir != '':
            load_path = self.args.load_dir + '/model.pt'
            o_mean, o_std, g_mean, g_std, model = torch.load(load_path)
            self.o_norm.mean = o_mean
            self.o_norm.std = o_std
            self.actor_network.load_state_dict(model)

        # sync the networks across the cpus
        # sync_networks(self.actor_network)
        # sync_networks(self.critic_network)
        # build up the target network
        # if self.image:
        #     self.actor_target_network = actor_image(env_params, env_params['obs'])
        # else:
        #     self.actor_target_network = actor(env_params, env_params['obs'])
        # # load the weights into the target networks
        # self.actor_target_network.load_state_dict(self.actor_network.state_dict())

        # if use gpu
        if self.args.cuda:
            self.actor_network.cuda()
            # self.actor_target_network.cuda()
            self.critic_network.cuda()
        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)
        # her sampler
        self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, self.env().compute_reward)
        # create the replay buffer
        self.buffer = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions, image=self.image)

        # path to save the model
        self.model_path = os.path.join(self.args.save_dir, self.args.env_name)

    def learn(self):
        """
        train the network

        """
        # start to collect samples
        for epoch in range(self.args.n_epochs):
            total_actor_loss, total_critic_loss, count = 0, 0, 0
            for _ in range(self.args.n_cycles):
                if self.image:
                    mb_obs, mb_ag, mb_g, mb_sg, mb_actions, mb_hidden, mb_image = [], [], [], [], [], [], []
                else:
                    mb_obs, mb_ag, mb_g, mb_sg, mb_actions, mb_hidden = [], [], [], [], [], []
                for _ in range(self.args.num_rollouts_per_mpi):
                    # reset the rollouts
                    if self.image:
                        ep_obs, ep_ag, ep_g, ep_sg, ep_actions, ep_hidden, ep_image = [], [], [], [], [], [], []
                    else:
                        ep_obs, ep_ag, ep_g, ep_sg, ep_actions, ep_hidden = [], [], [], [], [], []
                    # reset the environment
                    e = self.env()
                    observation = e.reset()
                    obs = observation['observation']
                    ag = observation['achieved_goal']
                    g = observation['desired_goal']
                    img = observation['image']
                    sg = np.zeros(4)
                    hidden = np.zeros(64)

                    if self.image:
                        image_tensor = torch.tensor(observation['image'], dtype=torch.float32).unsqueeze(0)
                        if self.args.cuda:
                            image_tensor = image_tensor.cuda()
                    # start to collect samples
                    for _ in range(self.env_params['max_timesteps']):
                        with torch.no_grad():
                            input_tensor = self._preproc_inputs(obs)
                            if self.image:
                                pi = self.actor_network(input_tensor, image_tensor).detach()
                            else:
                                pi = self.actor_network(input_tensor).detach()
                            action = self._select_actions(pi, observation)
                        # feed the actions into the environment
                        observation_new, _, _, info = e.step(action)
                        obs_new = observation_new['observation']
                        ag_new = observation_new['achieved_goal']
                        img_new = observation_new['image']
                        # append rollouts
                        ep_obs.append(obs.copy())
                        ep_ag.append(ag.copy())
                        ep_g.append(g.copy())
                        ep_sg.append(sg.copy())
                        ep_actions.append(action.copy())
                        ep_hidden.append(hidden.copy())
                        if self.image:
                            ep_image.append(img.copy())
                        # re-assign the observation
                        obs = obs_new
                        ag = ag_new
                        img = img_new
                        observation = observation_new
                    ep_obs.append(obs.copy())
                    ep_ag.append(ag.copy())
                    ep_sg.append(sg.copy())
                    ep_hidden.append(hidden.copy())
                    if self.image:
                        ep_image.append(img.copy())
                    mb_obs.append(ep_obs)
                    mb_ag.append(ep_ag)
                    mb_g.append(ep_g)
                    mb_sg.append(ep_sg)
                    mb_actions.append(ep_actions)
                    mb_hidden.append(ep_hidden)
                    if self.image:
                        mb_image.append(ep_image)

                # convert them into arrays
                mb_obs = np.array(mb_obs)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                mb_sg = np.array(mb_sg)
                mb_actions = np.array(mb_actions)
                mb_hidden = np.array(mb_hidden)
                if self.image:
                    mb_image = np.array(mb_image)
                    self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions, mb_sg, mb_hidden, mb_image])
                # store the episodes
                else:
                    self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions, mb_sg, mb_hidden])
                self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])

                # train the network
                actor_loss, critic_loss = self._update_network()
                total_actor_loss += actor_loss
                total_critic_loss += critic_loss
                count += 1

                # soft update
                # self._soft_update_target_network(self.actor_target_network, self.actor_network)
            # start to do the evaluation
            # success_rate = self._eval_agent()
            # print('[{}] epoch is: {}, actor loss is: {:.5f}, critic loss is: {:.5f}, eval success rate is: {:.3f}'.format(
            #     datetime.now(), epoch, total_actor_loss/count, total_critic_loss/count, success_rate))

            if self.args.cuda:
                torch.cuda.empty_cache()
            torch.save([self.o_norm.mean, self.o_norm.std, self.actor_network.state_dict()], \
                        self.model_path + '/actor.pt')
            torch.save([self.o_norm.mean, self.o_norm.std, self.critic_network.state_dict()], \
                        self.model_path + '/critic.pt')

    # pre_process the inputs
    def _preproc_inputs(self, obs):
        obs_norm = self.o_norm.normalize(obs)
        # concatenate the stuffs
        inputs = obs_norm
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
        return inputs

    # this function will choose action for the agent and do the exploration
    def _select_actions(self, pi, observation):
        action = pi.cpu().numpy().squeeze()
        # add the gaussian
        action += self.args.noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])
        # random actions...
        # random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
        #                                     size=self.env_params['action'])
        offset = (observation["achieved_goal"] - observation["gripper_pose"])
        # if observation['observation'][-1] < 1:
        offset /= 0.05
        offset += self.args.noise_eps * self.env_params['action_max'] * np.random.randn(*offset.shape)
        offset *= 0.05
        offset /= np.random.uniform(0.03, 0.07)
        # else:
        #     offset /= np.random.uniform(0.03, 0.07)
        #     offset += self.args.noise_eps * self.env_params['action_max'] * np.random.randn(*offset.shape)
        good_action = np.clip(np.array([offset[0], offset[1], offset[2], 1.]), -self.env_params['action_max'], self.env_params['action_max'])
        # choose if use the random actions
        if np.random.uniform() < self.args.random_eps:
            action = good_action
        return action

    # update the normalizer
    def _update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs,
                       'ag': mb_ag,
                       'g': mb_g,
                       'actions': mb_actions,
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       }
        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
        obs = transitions['obs']
        # pre process the obs and g
        transitions['obs'] = self._preproc_og(obs)
        # update
        self.o_norm.update(transitions['obs'])
        # recompute the stats
        self.o_norm.recompute_stats()

    def _preproc_og(self, o):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        return o

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    # update the network
    def _update_network(self):
        # profiling
        start = time.time()

        # sample the episodes
        transitions = self.buffer.sample(self.args.batch_size)
        # pre-process the observation and goal
        o, o_next = transitions['obs'], transitions['obs_next']
        input_tensor = torch.tensor(o, dtype=torch.float32)
        counter = 1-input_tensor[:, -1:]
        input_next_tensor = torch.tensor(o_next, dtype=torch.float32)
        transitions['obs'] = self._preproc_og(o)
        transitions['obs_next'] = self._preproc_og(o_next)
        # start to do the update
        obs_norm = self.o_norm.normalize(transitions['obs'])
        inputs_norm = obs_norm
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        inputs_next_norm = obs_next_norm
        # print("avg rewards {}".format(np.mean(transitions['r'])))

        # profiling
        now = time.time()
        print("getting numpy data from storage: {}".format(now-start))
        start = now

        # transfer them into the tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32)
        box_tensor = torch.tensor(transitions['ag'], dtype=torch.float32)
        box_next_tensor = torch.tensor(transitions['ag_next'], dtype=torch.float32)
        if self.image:
            img_tensor = torch.tensor(transitions['image'], dtype=torch.float32)
            img_next_tensor = torch.tensor(transitions['image_next'], dtype=torch.float32)

        # profiling
        now = time.time()
        print("numpy to tensor: {}".format(now-start))
        start = now

        if self.args.cuda:
            inputs_norm_tensor = inputs_norm_tensor.cuda()
            inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()
            box_tensor = box_tensor.cuda()
            box_next_tensor = box_next_tensor.cuda()
            input_tensor = input_tensor.cuda()
            input_next_tensor = input_next_tensor.cuda()
            counter = counter.cuda()
            if self.image:
                img_tensor = img_tensor.cuda()
                img_next_tensor = img_next_tensor.cuda()

        # profiling
        now = time.time()
        print("move to cuda: {}".format(now-start))
        start = now

        # calculate the target Q value function
        with torch.no_grad():
            # do the normalization
            # concatenate the stuffs
            # if self.image:
            #     actions_next = self.actor_target_network(inputs_next_norm_tensor, img_next_tensor)
            # else:
            #     actions_next = self.actor_target_network(inputs_next_norm_tensor)
            q_next_value = next_q_estimator(input_next_tensor, box_next_tensor)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + self.args.gamma * q_next_value * counter
            # print("expected q value {}".format(target_q_value.mean().item()))
            # print(r_tensor, q_next_value, counter)
            target_q_value = target_q_value.detach()
            # print(torch.masked_select(target_q_value, mask), torch.masked_select(r_tensor, mask))
            # clip the q value
            clip_return = 1 / (1 - self.args.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, clip_return)

        # profiling
        now = time.time()
        print("get target q value: {}".format(now-start))
        start = now

        # the q loss
        if self.image:
            real_q_value = self.critic_network(inputs_norm_tensor, img_tensor, actions_tensor)
        else:
            real_q_value = self.critic_network(inputs_norm_tensor, actions_tensor)
        # print(target_q_value, real_q_value, input_tensor[:, :3], actions_tensor[:, :3], box_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()

        # profiling
        now = time.time()
        print("compute q: {}".format(now-start))
        start = now

        self.critic_optim.zero_grad()
        critic_loss.backward()
        # sync_grads(self.critic_network)
        self.critic_optim.step()
        critic_loss_value = critic_loss.detach().item()

        # profiling
        now = time.time()
        print("update critic: {}".format(now-start))
        start = now

        # the actor loss
        if self.image:
            actions_real = self.actor_network(inputs_norm_tensor, img_tensor)
            actor_loss = -self.critic_network(inputs_norm_tensor, img_tensor, actions_real).mean()
        else:
            actions_real = self.actor_network(inputs_norm_tensor)
            actor_loss = -self.critic_network(inputs_norm_tensor, actions_real).mean()
        actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()

        # profiling
        now = time.time()
        print("compute actor loss: {}".format(now-start))
        start = now

        # start to update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        # sync_grads(self.actor_network)
        self.actor_optim.step()
        actor_loss_value = actor_loss.detach().item()

        # profiling
        now = time.time()
        print("actor update: {}".format(now-start))
        start = now

        if self.args.cuda:
            torch.cuda.empty_cache()

        # profiling
        now = time.time()
        print("cleaning cache: {}".format(now-start))
        start = now

        return actor_loss_value, critic_loss_value

    # do the evaluation
    def _eval_agent(self):
        total_success_rate = []
        for _ in range(self.args.n_test_rollouts):
            e = self.env()
            observation = e.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            if self.image:
                img_tensor = torch.tensor(observation['image'], dtype=torch.float32).unsqueeze(0)
                if self.args.cuda:
                    img_tensor = img_tensor.cuda()
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs)
                    if self.image:
                        pi = self.actor_network(input_tensor, img_tensor)
                    else:
                        pi = self.actor_network(input_tensor)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()
                observation, _, _, info = e.step(actions)
                obs = observation['observation']
                g = observation['desired_goal']
                if self.image:
                    img_tensor = torch.tensor(observation['image'], dtype=torch.float32).unsqueeze(0)
                    if self.args.cuda:
                        img_tensor = img_tensor.cuda()
            total_success_rate.append(info['is_success'])
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate)
        return local_success_rate
