'''
Training script for quadruped robot that supports both ARS and PPO algorithms.
'''

import argparse
import ray
from pupper_ars_train import run_ars
from pupper_ppo_train import run_ppo

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, choices=['ars', 'ppo'], default='ars',
                      help='Algorithm to use for training (ars or ppo)')
    
    # Common arguments
    parser.add_argument('--n_iter', '-n', type=int, default=1000)
    parser.add_argument('--n_workers', '-e', type=int, default=32)
    parser.add_argument('--dir_path', type=str, default='data')
    
    # ARS-specific arguments
    parser.add_argument('--n_directions', '-nd', type=int, default=16)
    parser.add_argument('--deltas_used', '-du', type=int, default=16)
    parser.add_argument('--step_size', '-s', type=float, default=0.03)
    parser.add_argument('--delta_std', '-std', type=float, default=.03)
    parser.add_argument('--rollout_length', '-r', type=int, default=400)
    parser.add_argument('--shift', type=float, default=0)
    parser.add_argument('--seed', type=int, default=37)
    parser.add_argument('--policy_type', type=str, default='linear')
    parser.add_argument('--filter', type=str, default='MeanStdFilter')
    parser.add_argument('--policy_network_size_list', type=str, default='64,64')
    parser.add_argument('--activation', type=str, default='tanh')
    
    # PPO-specific arguments
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--clip_ratio', type=float, default=0.2)
    parser.add_argument('--value_coef', type=float, default=0.5)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.95)
    
    args = parser.parse_args()
    params = vars(args)
    
    ray.init()
    assert ray.is_initialized()
    try:
        if args.algorithm == 'ars':
            run_ars(params)
        else:
            run_ppo(params)
    finally:
        ray.shutdown()

if __name__ == '__main__':
    main() 