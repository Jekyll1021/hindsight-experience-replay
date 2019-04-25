import torch
import os
from datetime import datetime
import numpy as np
from mpi4py import MPI
from models import open_loop_image_predictor
from utils import sync_networks, sync_grads
from open_loop_buffer import open_loop_buffer

"""
ddpg with HER (MPI-version)

"""
class open_loop_agent:
    def __init__(self, args, env, env_params, sample_size=100):
        self.args = args
        self.env = env
        self.env_params = env_params
        self.sample_size = sample_size

        # create the network
        self.score_predictor = open_loop_image_predictor(env_params['obs'] + env_params['action'])

        # load model if load_path is not None
        if self.args.load_dir != '':
            load_path = self.args.load_dir + '/model.pt'
            o_mean, o_std, a_mean, a_std, model = torch.load(load_path)
            self.o_norm.mean = o_mean
            self.o_norm.std = o_std
            self.a_norm.mean = a_mean
            self.a_norm.std = a_std
            self.score_predictor.load_state_dict(model)

        # sync the networks across the cpus
        # sync_networks(self.score_predictor)

        # if use gpu
        if self.args.cuda:
            self.score_predictor.cuda()

        # create the optimizer
        self.optim = torch.optim.Adam(self.score_predictor.parameters(), lr=self.args.lr_actor)

        # create the replay buffer
        self.buffer = open_loop_buffer(self.env_params, self.args.buffer_size)

        # path to save the model
        self.model_path = os.path.join(self.args.save_dir, self.args.env_name)

    def learn(self):
        """
        train the network

        """
        # start to collect samples
        for epoch in range(self.args.n_epochs):
            mb_obs, mb_actions, mb_success, mb_image = [], [], [], []
            count = 0
            failure = int(self.sample_size * 2 / 3)
            while count < self.sample_size:
                # reset the environment
                observation = self.env.reset()
                obs = observation['observation']
                image = observation['image']
                # start to collect samples
                sample_mean = observation['achieved_goal'][:2]
                if failure > 0:
                    sample_cov = np.eye(self.env_params['action']) * 0.03
                else:
                    sample_cov = np.eye(self.env_params['action']) * 0.01
                action = np.random.multivariate_normal(sample_mean, sample_cov)
                _, _, _, info = self.env.step(action)

                if info['is_success'] or failure > 0:
                    mb_obs.append(obs.copy())
                    mb_actions.append(action.copy())
                    mb_success.append([info['is_success']])
                    mb_image.append(image.copy())
                    failure -= (1 - info['is_success'])
                    count += 1

            # convert them into arrays
            mb_obs = np.array(mb_obs)
            mb_actions = np.array(mb_actions)
            mb_success = np.array(mb_success)
            mb_image = np.array(mb_image)

            # store the episodes
            self.buffer.store_episode([mb_obs, mb_actions, mb_success, mb_image])
            for _ in range(self.args.n_batches):
                # train the network
                loss = self._update_network()
            # start to do the evaluation
            success_rate = self._eval_agent()
            if MPI.COMM_WORLD.Get_rank() == 0:
                print('[{}] epoch {}: loss {:.5f}, eval success rate {:.3f}'.format(datetime.now(), epoch, loss, success_rate))
                torch.save(self.score_predictor.state_dict(), \
                            self.model_path + '/model.pt')

    # pre_process the inputs
    def _preproc_inputs(self, obs, action):
        # concatenate the stuffs
        inputs = np.concatenate([obs, action])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
        return inputs

    def _preproc_batch_inputs(self, obs, action):
        # concatenate the stuffs
        inputs = np.concatenate([obs, action], axis=1)
        inputs = torch.tensor(inputs, dtype=torch.float32)
        if self.args.cuda:
            inputs = inputs.cuda()
        return inputs

    # update the network
    def _update_network(self):
        # sample the episodes
        transitions = self.buffer.sample(self.args.batch_size)

        # start to do the update
        inputs_norm = np.concatenate([transitions['obs'], transitions['actions']], axis=1)

        # transfer them into the tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        image_tensor = torch.tensor(transitions['image'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['success'], dtype=torch.float32)
        if self.args.cuda:
            inputs_norm_tensor = inputs_norm_tensor.cuda()
            image_tensor = image_tensor.cuda()
            r_tensor = r_tensor.cuda()

        score = self.score_predictor(inputs_norm_tensor, image_tensor)
        loss = torch.nn.functional.binary_cross_entropy(score, r_tensor)
        # start to update the network
        self.optim.zero_grad()
        loss.backward()
        # sync_grads(self.score_predictor)
        self.optim.step()
        # update the critic_network
        return loss


    # do the evaluation
    def _eval_agent(self):
        total_success_rate = []
        for _ in range(self.args.n_test_rollouts):
            observation = self.env.reset()
            obs = np.array([observation['observation']])
            image = np.array([observation['image']])
            # start to collect samples
            sample_mean = observation['achieved_goal'][:2]
            sample_cov = np.eye(self.env_params['action']) * 0.03
            a1 = np.random.multivariate_normal(sample_mean, sample_cov, 700)
            sample_cov = np.eye(self.env_params['action']) * 0.01
            a2 = np.random.multivariate_normal(sample_mean, sample_cov, 300)
            action = np.concatenate((a1, a2), axis=0)
            obs = np.repeat(obs, 1000, axis=0)
            image_tensor = torch.tensor(np.repeat(image, 1000, axis=0), dtype=torch.float32)
            if self.args.cuda:
                image_tensor = image_tensor.cuda()
            with torch.no_grad():
                input_tensor = self._preproc_batch_inputs(obs, action)
                score = self.score_predictor(input_tensor, image_tensor)
                ind = torch.argmax(score).item()

            _, _, _, info = self.env.step(action[ind])
            total_success_rate.append(info['is_success'])

        local_success_rate = np.mean(total_success_rate)
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size()
