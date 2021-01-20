import argparse
import pickle
import time

import numpy as np
import pandas as pd

from drl_bis.agent import DQNAgent
from drl_bis.environment import MultiStockEnv
from drl_bis.utils import check_dir, get_scaler, play_one_episode, elapsed_time
# Todo: pull out positions for every epoch

def get_data():
    # returns a T x 3 list of stock prices
    # each row is a different stock
    # 0 = AAPL
    # 1 = MSI
    # 2 = SBUX
    df = pd.read_csv('data/aapl_msi_sbux.csv', dtype=np.float32)
    return df.to_numpy()


if __name__ == '__main__':
    # config
    models_folder: str = 'rl_trader_models'
    rewards_folder: str = 'rl_trader_rewards'
    envs_folder: str = 'rl_trader_cps'
    check_dir(models_folder)
    check_dir(rewards_folder)
    check_dir(envs_folder)

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True, help='Mode: "train" or "test".')
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help="Batch size.")
    parser.add_argument('-e', '--num_episodes', type=int, default=2_000, help="Number of episodes.")
    parser.add_argument('--epsilon', type=float, default=0.01, help="Epsilon-greedy value used for 'test' mode.")
    parser.add_argument('-i', '--init_investment', type=float, default=20_000.0, help="Initial investments.")
    parser.add_argument('-s', '--train_size', type=float, default=0.5,
                        help="Train test split (specify training proportion).")
    parser.add_argument('-l', '--load', type=str, default=None,
                        help="Experiment label (for test mode). E.g., 'tue_19_jan_2021_12h18_i20000_s80_bs32_e5'")
    args = parser.parse_args()

    batch_size: int = int(args.batch_size)
    num_episodes: int = int(args.num_episodes)
    initial_investment: float = float(args.init_investment)

    data: np.ndarray = get_data()
    n_timesteps, n_stocks = data.shape

    n_train: int = int(n_timesteps * args.train_size)

    experiment_label: str = time.strftime("%a_%d_%b_%Y_%Hh%M", time.localtime()).lower()
    experiment_label += f"_i{initial_investment:.0f}"
    experiment_label += f"_s{int(args.train_size * 100)}"
    experiment_label += f"_bs{batch_size}_e{num_episodes}"
    experiment_label += f"_{args.mode}"

    print('\n' + '*' * 50)
    print(' ' * 18 + 'CONFIGURATION')
    print('-' * 50)
    print(f"Experiment {experiment_label}")
    print(f"Mode: {args.mode}")
    print(f"Batch size: {batch_size}")
    print(f"Number of episodes: {num_episodes:,}")
    print(f"Initial investment: {initial_investment:,}")
    print(f"Training size: {int(args.train_size * 100)}%, that is, {n_train:,} time periods.")
    print(f"Data shape: {data.shape[0]} periods, and {data.shape[1]} stocks.")
    print('*' * 50 + "\n")

    train_data = data[:n_train]
    test_data = data[n_train:]

    env = MultiStockEnv(train_data, initial_investment)
    state_size = env.state_dim
    action_size = len(env.action_space)
    agent = DQNAgent(state_size, action_size)
    scaler = get_scaler(env)

    # store the final value of the portfolio (end of episode)
    portfolio_value = []

    if args.mode == 'test':
        # then load the previous scaler
        with open(f'{models_folder}/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        # remake the env with test data
        env = MultiStockEnv(test_data, initial_investment)

        # make sure epsilon is not 1!
        # no need to run multiple episodes if epsilon = 0, it's deterministic
        agent.epsilon = args.epsilon

        # load trained weights
        agent.load(f'{models_folder}/dqn.h5')

    # play the game num_episodes times
    for e in range(num_episodes):
        start: float = time.time()
        val: float = play_one_episode(agent, env, is_train=args.mode, batch_size=batch_size, scaler=scaler)
        duration = elapsed_time(start, time.time())
        print(f"Episode: {e + 1}/{num_episodes}, Episode return (portfolio value): {val:.2f}, {duration}")
        portfolio_value.append(val)  # append episode end portfolio value

    # save the weights when we are done
    if args.mode == 'train':
        # save the DQN
        out_agent_file: str = f'{models_folder}/dqn-{experiment_label}.h5'
        agent.save(out_agent_file)
        print(f"Successfully saved agent: {out_agent_file!r}")

        # save the scaler
        with open(f'{models_folder}/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

    # save portfolio value for each episode
    out_rewards_file: str = f'{rewards_folder}/dqn-{experiment_label}.npy'
    np.save(out_rewards_file, portfolio_value)
    print(f"Successfully saved rewards: {out_rewards_file!r}")

    # This only holds for the last epoch
    out_envs_file: str = f'dqn-{experiment_label}.npy'
    env.save(out_envs_file, path=envs_folder, verbose=1)
