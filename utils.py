from pathlib import Path
from typing import Optional

from cycler import cycler
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {'blue': '#6370F1',
          'orange': '#F3A467',
          'green': '#59C899',
          'purple': '#A16AF2',
          'light_blue': '#5BCCC7',
          'red': '#DF6046'}

CYCLER = cycler(color=COLORS.values())
LEGEND_PARAMS = dict(loc='center', bbox_to_anchor=(0.5, 1.05), edgecolor='none',
                     fancybox=True, framealpha=1, facecolor='none')


def get_data(out_as: str = 'np'):
    # returns a T x 3 list of stock prices
    # each row is a different stock
    # 0 = AAPL
    # 1 = MSI
    # 2 = SBUX
    df = pd.read_csv('data/aapl_msi_sbux.csv', dtype=np.float32)
    if out_as in ('np', 'numpy'):
        return df.to_numpy()
    elif out_as in ('pd', 'pandas'):
        return df


def saveplot(figname: str, fig_path: Optional[Path] = None):
    if figname is None:
        raise TypeError("You must pass a 'str' for the figure name.")
    filename = f'{figname}.pdf' if fig_path is None else fig_path / f'{figname}.pdf'
    plt.savefig(filename, dpi=600, transparent=True)
    print(f"Figure successfully saved {filename}")


def plot_hist(series, fig_path: Path = None, save: bool = False, figname: Optional[str] = None):
    plt.hist(series, bins=10, edgecolor='k')
    plt.ylabel('frequency')
    plt.xlabel('normalized average portfolio rewards')
    plt.tight_layout()
    if save:
        saveplot(figname, fig_path)
    plt.show()


def plot_positions(data: pd.DataFrame,
                   fig_path: Optional[Path] = None,
                   figname: Optional[str] = None,
                   save: bool = False):
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"You have to pass a pandas DataFrame, not a {type(data).__name__!r}")
    cmap = cycler(color=['#ef476f', '#ffd166', '#06d6a0', '#118ab2'])
    x = range(len(data))
    data_pct = data.divide(data.sum(axis=1), axis=0).mul(100)
    col_list = [data_pct[col] for col in data_pct]

    plt.rc('axes', prop_cycle=cmap)
    plt.figure(figsize=(6.4, 4.8))
    plt.stackplot(x, *col_list)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.ylabel('positions')
    plt.xlabel('time steps')
    plt.legend(**LEGEND_PARAMS, ncol=4, labels=data_pct.columns)
    plt.tight_layout()
    if save:
        saveplot(figname, fig_path)
    plt.show()
