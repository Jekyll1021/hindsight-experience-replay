import numpy as np
import gym
import os, sys
from arguments import get_args
from mpi4py import MPI
from subprocess import CalledProcessError
from ddpg_joint_agent import ddpg_joint_agent
from ddpg_joint_ee_agent import ddpg_joint_ee_agent

"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)

"""
def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'action': env.action_space.shape[0],
            'action_max': env.action_space.high[0],
            }
    params['max_timesteps'] = env._max_episode_steps
    return params

def launch(args):
    env_name_lst = [
        'CamPush-v0',
        # 'CamReach-v0',
        # 'CamPickAndPlace-v0'
    ]
    expert_model_path_lst = [
        '../hindsight-experience-replay/saved_models/CamPush-v0',
        # '../hindsight-experience-replay/saved_models/CamReach-v0',
        # '../hindsight-experience-replay/saved_models/CamPickAndPlace-v0'
    ]
    # create the ddpg_agent
    env_lst = [gym.make(name, reward_type='sparse', goal_type='random', cam_type='fixed', gripper_init_type='random', act_noise=False, obs_noise=False) for name in env_name_lst]
    # get the environment parameters
    env_params = get_env_params(env_lst[0])
    # create the ddpg agent to interact with the environment
    ddpg_trainer = ddpg_joint_agent(args, env_lst, env_params, expert_model_path_lst)
    ddpg_trainer.learn()

if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    # get the params
    args = get_args()
    if MPI.COMM_WORLD.Get_rank() == 0:
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
        # path to save the model
        model_path = os.path.join(args.save_dir, args.env_name)
        if not os.path.exists(model_path):
            os.mkdir(model_path)
    launch(args)
