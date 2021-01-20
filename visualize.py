import matplotlib.pyplot as plt
import numpy as np
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='train', help="Modes: 'train' or 'test")
    args = parser.parse_args()

    r = np.load(f'rl_trader_rewards/{args.mode}.npy')
    print(f"Average reward: {r.mean():.2f}, min: {r.min():.2f}, max: {r.max():.2f}")

    plt.hist(r, bins=20)
    plt.title(f"{args.mode}")
    plt.show()
