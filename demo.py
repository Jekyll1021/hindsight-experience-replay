import torch
from models import actor, actor_image, critic, critic_image
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
    use_expert = args.use_expert
    # load the model param
    actor_path = args.save_dir + args.env_name + '/actor.pt'
    critic_path = args.save_dir + args.env_name + '/critic.pt'
    # o_mean, o_std, g_mean, g_std, model = torch.load(model_path, map_location=lambda storage, loc: storage)
    o_mean, o_std, actor_model = torch.load(actor_path, map_location=lambda storage, loc: storage)
    o_mean, o_std, critic_model = torch.load(critic_path, map_location=lambda storage, loc: storage)
    # create the environment
    env = gym.make(args.env_name, reward_type='sparse', goal_type='random', cam_type='fixed', gripper_init_type='fixed', act_noise=False, obs_noise=False, two_cam=False)
    # get the env param
    observation = env.reset()
    # get the environment params
    env_params = {'obs': observation['observation'].shape[0],
                  'goal': observation['desired_goal'].shape[0],
                  'action': env.action_space.shape[0],
                  'action_max': env.action_space.high[0],
                  'depth': env.env.depth,
                  'two_cam': env.env.two_cam
                  }
    # create the actor network
    if use_image:
        actor_network = actor_image(env_params, env_params['obs'])
        critic_network = critic_image(env_params, env_params['obs'] + env_params['action'])
    else:
        actor_network = actor(env_params, env_params['obs'])
        critic_network = critic_image(env_params, env_params['obs'] + env_params['action'])
    actor_network.load_state_dict(actor_model)
    actor_network.eval()
    critic_network.load_state_dict(critic_model)
    critic_network.eval()

    total_success_rate = []
    for i in range(args.demo_length):
        path_ind = os.path.join(path, str(i))
        os.makedirs(path_ind)
        env = gym.make(args.env_name, reward_type='sparse', goal_type='random', cam_type='fixed', gripper_init_type='fixed', act_noise=False, obs_noise=False, two_cam=False)
        # get the env param
        observation = env.reset()
        # start to do the demo
        obs = observation['observation']
        g = observation['desired_goal']
        img = observation['image']
        if env_params['two_cam']:
            img1, img2 = np.array_split(img, 2, axis=-1)
        for t in range(env._max_episode_steps):
            demo_img = env.render('rgb_array')
            # inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std, args)
            if env_params['two_cam']:
                cv2.imwrite(os.path.join(path_ind, str(t)+"_1.png"), img1*255)
                cv2.imwrite(os.path.join(path_ind, str(t)+"_2.png"), img2*255)
            else:
                cv2.imwrite(os.path.join(path_ind, str(t)+".png"), img*255)
            inputs = process_inputs(obs, o_mean, o_std, args)
            with torch.no_grad():
                if use_image:
                    image_tensor = torch.tensor([img], dtype=torch.float32)
                    pi = actor_network(inputs, image_tensor)

                    offset = (observation["achieved_goal"] - observation["gripper_pose"])
                    offset /= 0.05
                    good_action = np.clip(np.array([offset[0], offset[1], offset[2], 1.]), -env_params['action_max'], env_params['action_max'])

                    if np.random.uniform() < args.random_eps:
                        print("use expert on step {}".format(t))
                        pi = torch.tensor(good_action, dtype=torch.float32)

                    print("offset: {}, actions: {}".format(((observation["achieved_goal"] - observation["gripper_pose"])/0.05).tolist(), (pi[:3]).tolist()))
                    q_value = critic_network(inputs, image_tensor, pi)
                else:
                    pi = actor_network(inputs)
                    q_value = critic_network(inputs, pi)
            action = pi.detach().numpy().squeeze()
            value = q_value.detach().item()
            print("rollout: {}, step: {}, q_value: {}".format(i, t, value))
            # put actions into the environment
            observation, reward, _, info = env.step(action)
            obs = observation['observation']
            img = observation['image']
            if env_params['two_cam']:
                img1, img2 = np.array_split(img, 2, axis=-1)
        if env_params['two_cam']:
            cv2.imwrite(os.path.join(path_ind, str(t+1)+"_1.png"), img1 * 255)
            cv2.imwrite(os.path.join(path_ind, str(t+1)+"_2.png"), img2 * 255)
        else:
            cv2.imwrite(os.path.join(path_ind, str(t+1)+".png"), img * 255)
        print('the episode is: {}, is success: {}'.format(i, info['is_success']))
        total_success_rate.append(info['is_success'])
    print('total success rate: {}'.format(np.sum(total_success_rate) / len(total_success_rate)))
