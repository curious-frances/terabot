"""

Code to load a policy and generate rollout data. Adapted from https://github.com/berkeleydeeprlcourse. 
Example usage:
python3 pupper_ars_run_policy.py --expert_policy_file=data/lin_policy_plus_latest.npz --json_file=data/params.json --render

"""
import numpy as np
import gym
import time
import os
import json
import pickle
import puppersim
import gin
from pybullet import COV_ENABLE_GUI
import puppersim.data as pd
import argparse
from puppersim.robot import PupperRobot

def create_pupper_env(args):
    if args.run_on_robot:
        # Create robot environment directly
        env = PupperRobot()
        return env
    else:
        # Import env_loader here to avoid pybullet_envs issues
        from pybullet_envs.minitaur.envs_v2 import env_loader
        
        CONFIG_DIR = puppersim.getPupperSimPath()
        _CONFIG_FILE = os.path.join(CONFIG_DIR, "config", "pupper_pmtg.gin")
        gin.bind_parameter("scene_base.SceneBase.data_root", pd.getDataPath()+"/")
        gin.parse_config_file(_CONFIG_FILE)
        gin.bind_parameter("SimulationParameters.enable_rendering", args.render)
        env = env_loader.load()
        env._pybullet_client.configureDebugVisualizer(COV_ENABLE_GUI, 0)
        return env

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert_policy_file', type=str, required=True,
                      help='Path to the policy file (.npz)')
    parser.add_argument('--json_file', type=str, required=True,
                      help='Path to the parameters JSON file')
    parser.add_argument('--run_on_robot', action='store_true',
                      help='whether to run the policy on the robot instead of in simulation. Default is False.')
    parser.add_argument('--render', default=False, action='store_true',
                      help='whether to render the robot. Default is False.')
    parser.add_argument('--profile', default=False, action='store_true',
                      help='whether to print timing for parts of the code. Default is False.')
    parser.add_argument('--plot', default=False, action='store_true',
                      help='whether to plot action and observation histories after running the policy.')
    parser.add_argument("--log_to_file", default=False, action='store_true',
                      help="Whether to log data to the disk.")
    parser.add_argument("--realtime", default=False, help="Run at realtime.")
    args = parser.parse_args(argv)

    print('loading and building expert policy')
    with open(args.json_file) as f:
        params = json.load(f)
    
    # Add env_name if not present
    if 'env_name' not in params:
        params['env_name'] = 'PupperEnv-v0'
    
    print("params=", params)
    
    # Create environment
    env = create_pupper_env(args)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Load policy
    policy = np.load(args.expert_policy_file)
    w = policy['w']
    
    returns = []
    observations = []
    actions = []

    log_dict = {
        't': [],
        'IMU': [],
        'MotorAngle': [],
        'action': []
    }

    try:
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        start_time_wall = time.time()
        env_start_time_wall = time.time()
        last_spammy_log = 0.0
        
        while not done or args.run_on_robot:
            if args.realtime or args.run_on_robot:  # always run at realtime with real robot
                # Sync to real time.
                wall_elapsed = time.time() - env_start_time_wall
                sim_elapsed = env.env_step_counter * env.env_time_step
                sleep_time = sim_elapsed - wall_elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                elif sleep_time < -1 and time.time() - last_spammy_log > 1.0:
                    print(f"Cannot keep up with realtime. {-sleep_time:.2f} sec behind, "
                          f"sim/wall ratio {(sim_elapsed/wall_elapsed):.2f}.")
                    last_spammy_log = time.time()

            if args.profile:
                print("loop dt:", time.time() - start_time_wall)
            start_time_wall = time.time()
            before_policy = time.time()
            
            # Get action from policy
            action = w.dot(obs)
            
            after_policy = time.time()

            if not args.run_on_robot:
                observations.append(obs)
                actions.append(action)

            next_obs, r, done, _ = env.step(action)
            
            if args.log_to_file:
                log_dict['t'].append(env.robot.GetTimeSinceReset())
                log_dict['MotorAngle'].append(next_obs[0:12])
                log_dict['IMU'].append(next_obs[12:16])
                log_dict['action'].append(action)

            totalr += r
            steps += 1
            obs = next_obs
            
            if args.profile:
                print('policy.act(obs): ', after_policy - before_policy)
                print('wallclock_control_code: ', time.time() - start_time_wall)
                
        returns.append(totalr)
    finally:
        if args.log_to_file:
            print("logging to file...")
            with open("env_ars_log.txt", "wb") as f:
                pickle.dump(log_dict, f)

    print('returns: ', returns)
    print('mean return: ', np.mean(returns))
    print('std of return: ', np.std(returns))

    if args.plot and not args.run_on_robot:
        import matplotlib.pyplot as plt
        action_history = np.array(actions)
        observation_history = np.array(observations)
        plt.plot(action_history)
        plt.show()

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
