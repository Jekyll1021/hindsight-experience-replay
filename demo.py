import torch
from models import actor, actor_image
from arguments import get_args
import gym
import numpy as np
import cv2

import os

# process the inputs
# def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
def process_inputs(o, o_mean, o_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    # g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
    # g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    # inputs = np.concatenate([o_norm, g_norm])
    inputs = np.array([o_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs

if __name__ == '__main__':
    path = "/checkpoint/jdong1021/visualization"
    os.makedirs(path)

    args = get_args()
    use_image = True
    # load the model param
    model_path = args.save_dir + args.env_name + '/model.pt'
    # o_mean, o_std, g_mean, g_std, model = torch.load(model_path, map_location=lambda storage, loc: storage)
    o_mean, o_std, model = torch.load(model_path, map_location=lambda storage, loc: storage)
    # create the environment
    env = gym.make(args.env_name, reward_type='sparse', goal_type='fixed', cam_type='fixed', gripper_init_type='fixed', act_noise=False, obs_noise=False)
    # get the env param
    observation = env.reset()
    # get the environment params
    env_params = {'obs': observation['observation'].shape[0],
                  'goal': observation['desired_goal'].shape[0],
                  'action': env.action_space.shape[0],
                  'action_max': env.action_space.high[0],
                  'depth': env.env.depth
                  }
    # create the actor network
    if use_image:
        actor_network = actor_image(env_params, env_params['obs'])
    else:
        actor_network = actor(env_params, env_params['obs'])
    actor_network.load_state_dict(model)
    actor_network.eval()
    for i in range(args.demo_length):
        path_ind = os.path.join(path, str(i))
        os.makedirs(path_ind)
        observation = env.reset()
        # start to do the demo
        obs = observation['observation']
        g = observation['desired_goal']
        img = observation['image']
        for t in range(env._max_episode_steps):
            demo_img = env.render('rgb_array')
            # inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std, args)
            cv2.imwrite(os.path.join(path_ind, str(t)+".png"), demo_img)
            inputs = process_inputs(obs, o_mean, o_std, args)
            with torch.no_grad():
                if use_image:
                    image_tensor = torch.tensor([img], dtype=torch.float32)
                    pi = actor_network(inputs, image_tensor)
                else:
                    pi = actor_network(inputs)
            action = pi.detach().numpy().squeeze()
            # put actions into the environment
            observation_new, reward, _, info = env.step(action)
            obs = observation_new['observation']
            img = observation_new['image']
        cv2.imwrite(path_ind+str(t)+".png", img * 255)
        print('the episode is: {}, is success: {}'.format(i, info['is_success']))
