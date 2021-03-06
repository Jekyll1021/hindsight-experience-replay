import threading
import numpy as np

"""
the replay buffer here is basically from the openai baselines code

"""
class open_loop_buffer:
    def __init__(self, env_params, buffer_size):
        self.env_params = env_params
        self.size = buffer_size
        # memory management
        self.current_size = 0
        # create the buffer to store info

        self.buffers = {'obs': np.empty([self.size, self.env_params['obs']]),
                        'actions': np.empty([self.size, self.env_params['action']]),
                        'success': np.empty([self.size, 1]),
                        'image': np.empty([self.size, 512, 512, 3])
                        }
        # thread lock
        self.lock = threading.Lock()

    # store the episode
    def store_episode(self, episode_batch):
        mb_obs, mb_actions, mb_success, mb_image = episode_batch
        sample_size = mb_obs.shape[0]
        with self.lock:
            idxs = self._get_storage_idx(inc=sample_size)
            # store the informations
            self.buffers['obs'][idxs] = mb_obs
            self.buffers['actions'][idxs] = mb_actions
            self.buffers['image'][idxs] = mb_image
            self.buffers['success'][idxs] = mb_success

    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size].copy()
        rollout_batch_size = temp_buffers['actions'].shape[0]
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        transitions = {key: temp_buffers[key][episode_idxs].copy() for key in temp_buffers.keys()}
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
