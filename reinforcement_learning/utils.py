import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib

current_cmap = matplotlib.cm.get_cmap()
current_cmap.set_bad(color='red')

def portfolio(returns, weights):
    weights = np.array(weights)
    rets = returns.mean()
    covs = returns.cov()
    P_ret = np.sum(rets * weights)
    P_vol = np.sqrt(np.dot(weights.T, np.dot(covs, weights)))
    P_sharpe = P_ret/P_vol
    return np.array([P_ret, P_vol, P_sharpe])

def plot_results(benchmark_series,
                 target_series,
                 target_balances,
                 n_assets,
                 columns,
                 name2plot = '',
                 path2save = './',
                 base_name_series = 'series'):

#     N = len(np.array(benchmark_series).cumsum())
    N = len(np.array([item for sublist in benchmark_series for item in sublist]).cumsum())

    if not os.path.exists(path2save):
        os.makedirs(path2save)

    for i in range(0, len(target_balances)):

        current_range = np.arange(0, N)
        current_ts = np.zeros(N)
        current_ts2 = np.zeros(N)

        ts_benchmark = np.array([item for sublist in benchmark_series[:i+1] for item in sublist]).cumsum()
        ts_target = np.array([item for sublist in target_series[:i+1] for item in sublist]).cumsum()

        t = len(ts_benchmark)
        current_ts[:t] = ts_benchmark
        current_ts2[:t] = ts_target

        current_ts[current_ts == 0] = ts_benchmark[-1]
        current_ts2[current_ts2 == 0] = ts_target[-1]

        plt.figure(figsize = (12, 10))

        plt.subplot(2, 1, 1)
        plt.bar(np.arange(n_assets), target_balances[i], color = 'grey')
        plt.xticks(np.arange(n_assets), columns, rotation='vertical')

        plt.subplot(2, 1, 2)
        plt.colormaps = current_cmap
        plt.plot(current_range[:t], current_ts[:t], color = 'black', label = 'Benchmark')
        plt.plot(current_range[:t], current_ts2[:t], color = 'red', label = name2plot)
        plt.plot(current_range[t:], current_ts[t:], ls = '--', lw = .1, color = 'black')
        plt.autoscale(False)
        plt.ylim([-1, 1])
        plt.legend()
        plt.savefig(path2save + base_name_series + str(i) + '.jpg')
