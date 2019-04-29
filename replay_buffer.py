import threading
import numpy as np

"""
the replay buffer here is basically from the openai baselines code

"""
class replay_buffer:
    def __init__(self, env_params, buffer_size, sample_func, ee_reward=False, image=False):
        self.env_params = env_params
        self.T = env_params['max_timesteps']
        self.size = buffer_size // self.T
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        self.sample_func = sample_func
        self.ee_reward = ee_reward
        self.image = image
        # create the buffer to store info
        if ee_reward:
            self.buffers = {'obs': np.empty([self.size, self.T + 1, self.env_params['obs']]),
                            'ag': np.empty([self.size, self.T + 1, self.env_params['goal']]),
                            'g': np.empty([self.size, self.T, self.env_params['goal']]),
                            'actions': np.empty([self.size, self.T, self.env_params['goal']]),
                            'sg': np.empty([self.size, self.T + 1, self.env_params['action']]),
                            'hidden': np.empty([self.size, self.T + 1, 64])
                            }
        else:
            self.buffers = {'obs': np.empty([self.size, self.T + 1, self.env_params['obs']]),
                            'ag': np.empty([self.size, self.T + 1, self.env_params['goal']]),
                            'g': np.empty([self.size, self.T, self.env_params['goal']]),
                            'actions': np.empty([self.size, self.T, self.env_params['action']]),
                            'sg': np.empty([self.size, self.T + 1, self.env_params['action']]),
                            'hidden': np.empty([self.size, self.T + 1, 64])
                            }
        if image:
            self.buffers["image"] = np.empty([self.size, self.T + 1, 256, 256, 3])
        # thread lock
        self.lock = threading.Lock()

    # store the episode
    def store_episode(self, episode_batch):
        if self.image:
            mb_obs, mb_ag, mb_g, mb_actions, mb_sg, mb_hidden, mb_image = episode_batch
        else:
            mb_obs, mb_ag, mb_g, mb_actions, mb_sg, mb_hidden = episode_batch
        batch_size = mb_obs.shape[0]
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            # store the informations
            self.buffers['obs'][idxs] = mb_obs
            self.buffers['ag'][idxs] = mb_ag
            self.buffers['g'][idxs] = mb_g
            self.buffers['sg'][idxs] = mb_sg
            self.buffers['actions'][idxs] = mb_actions
            self.buffers['hidden'][idxs] = mb_hidden
            self.n_transitions_stored += self.T * batch_size
            if self.image:
                self.buffers['image'][idxs] = mb_image

    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size].copy()
        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
        temp_buffers['sg_next'] = temp_buffers['sg'][:, 1:, :]
        temp_buffers['hidden_next'] = temp_buffers['hidden'][:, 1:, :]
        if self.image:
            temp_buffers['image_next'] = temp_buffers['image'][:, 1:, :]
        # sample transitions
        if self.ee_reward:
            transitions = self.sample_func(temp_buffers, batch_size, info="precise")
        else:
            transitions = self.sample_func(temp_buffers, batch_size)
        return transitions

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size+inc)
        if inc == 1:
            idx = idx[0]
        return idx
