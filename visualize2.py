from pathlib import Path

import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from drl_bis.agent import DQNAgent
from drl_bis.utils import check_dir


if __name__ == '__main__':
    SAVEFIG: bool = True
    check_dir('drl_bis/plots')
    path = Path('drl_bis/plots').resolve()
    # agent = DQNAgent(7, 27)
    # agent.load('rl_trader_models/dqn-tue_19_jan_2021_15h08_i20000_s80_bs32_e1.h5')

    r = np.load(f'rl_trader_rewards/test.npy')
    cps = np.load(f'rl_trader_cps/cps-dqn-tue_19_jan_2021_15h45_i20000_s80_bs32_e1.npy')
    so = np.load(f'rl_trader_cps/stocks_owned-dqn-tue_19_jan_2021_15h45_i20000_s80_bs32_e1.npy')

    legend_params = dict(loc='center', bbox_to_anchor=(0.5, 1.05), edgecolor='none',
                         fancybox=True, ncol=4, framealpha=1, facecolor='none')
    x = range(len(cps))
    cps = pd.DataFrame(cps)
    cp_pct = cps.divide(cps.sum(axis=1), axis=0).mul(100)

    plt.figure(figsize=(6.4, 4.8))
    plt.stackplot(x, cp_pct.iloc[:, 0], cp_pct.iloc[:, 1], cp_pct.iloc[:, 2], cp_pct.iloc[:, 3],
                  labels=('aapl', 'msci', 'sbux', 'bank account'),
                  colors=('#ef476f', '#ffd166', '#06d6a0', '#118ab2'))
    ax = plt.gca()
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.ylabel('positions')
    plt.xlabel('time steps')
    plt.legend(**legend_params)
    plt.tight_layout()
    if SAVEFIG:
        plt.savefig(path / 'training_weigths.pdf', dpi=600, transparent=True)
    plt.show()