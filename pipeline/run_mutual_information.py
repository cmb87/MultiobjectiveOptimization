import argparse
import os
from time import process_time

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from mutual_information import mutual_information
from util import get_starts, get_tags, get_logger


#################################################################


def calc_mi(data, tags, tag_list, config, output_dir):
    """
    """
    target = config['target']
    time_lags = config['time_lags']
    grad_periods = config['grad_periods']

    y = data[target]
    result = []

    time_lags_td = [pd.Timedelta(s) for s in time_lags]

    for i, s in enumerate(tags):

        if i % 10 == 0:
            print('Processing sensor %d / %d' % (i, len(tags)))

        x = data[s]
        mi = mutual_information(x.values, y.values, percentile_range=(0.1, 99.9))
        try:
            description = tag_list.loc[s].item()
        except KeyError:
            description = ''

        result.append((s, None, None, mi, description))

        for lag, dt in zip(time_lags, time_lags_td):
            x_lagged = x[x.index + dt]
            mi = mutual_information(x_lagged.values, y.values, percentile_range=(0.1, 99.9))
            result.append((s, lag, None, mi, description))

        for n in grad_periods:
            x_resampled = x.resample('1s').last()
            x_grad = x_resampled.diff(periods=n)
            x_grad = x_grad[y.index]
            mi = mutual_information(x_grad.values, y.values, percentile_range=(0.1, 99.9))
            result.append((s, None, n, mi, description))

    result = pd.DataFrame.from_records(result,
                                       columns=['tag', 'lag', 'grad', 'mi', 'description']).set_index('tag')
    result.sort_values('mi', ascending=False, inplace=True)

    result.to_csv(os.path.join(output_dir, 'mi.csv'), index=True)
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax.plot(range(len(result)), result['mi'].values)
    ax.set_title('Mutual information to %s, %d samples' % (target, len(data.index)))
    ax.grid()
    ax.set_xlabel('# inputs')
    ax.set_ylabel('mutual information to ' + target)
    plt.savefig(os.path.join(output_dir, 'mi.png'), dpi=100)
    return result


def main(config_file):
    with open(config_file) as yaml_config_file:
        config = yaml.load(yaml_config_file, Loader=yaml.FullLoader)

    output_dir = os.path.normpath(
        config['output_dir'] + '/' + os.path.splitext(os.path.basename(config_file))[0])
    os.makedirs(output_dir, exist_ok=True)
    logger = get_logger(os.path.join(output_dir, config['log_file']))
    logger.info('config file: ' + config_file)
    logger.info('output folder: ' + output_dir)

    tag_list = pd.read_csv(config['tag_list'], sep='\t', header=None, names=['tag', 'description'])
    tag_list.set_index('tag', inplace=True, verify_integrity=True)

    with pd.HDFStore(config['h5_file']) as hdf:
        data = []
        datasets = config['datasets']
        if len(datasets) == 0:
            datasets = hdf.keys()

        rpm_range = config['rpm_range']
        for key in datasets:
            logger.info('Loading dataset "%s" from file "%s"' % (key, config['h5_file']))
            df = hdf.get(key=key)

            if 'time_range' in config:
                rpm = df['Speed (rpm)']
                time_range = config['time_range']
                t0, use = get_starts(rpm, time_range=time_range)
                logger.info('Found %d starts in %s' % (len(t0), key))
                df = df[use]
                logger.info('Selected %d samples from %s around each start.' % (len(df.index), str(time_range)))

            rpm = df['Speed (rpm)']
            df = df[(rpm > rpm_range[0]) & (rpm < rpm_range[1])]
            logger.info('Selected %d samples with Speed (rpm) in %s' % (len(df.index), str(rpm_range)))

            df['dataset'] = key
            data.append(df)

    data = pd.concat(data, axis=0, sort=True)

    tags = get_tags(data, config['min_perc'])
    logger.info('Using %d / %d input features with at least %.0f%% defined values' % (len(tags),
                                                                                      len(data.columns),
                                                                                      config['min_perc'] * 100.0))
    t0 = process_time()
    result = calc_mi(data, tags, tag_list, config, output_dir)
    logger.info('Entropy calculations took %.1f s' % (process_time() - t0))

    # Omit Amplitude as self-reference in the following visualization
    target = config['target']
    result_wo_target = result[result.index != target]

    # Color-code the origin (file) of each sample, mapping each string to an int
    color = [datasets.index(s) for s in data['dataset']]
    fig, ax = plt.subplots(3, 5, figsize=(25, 15))
    ax = ax.ravel()
    for i in range(15):
        row = result_wo_target.iloc[i]  # Series
        tag = row.name
        lag = row.lag
        x = data[tag]
        if lag is not None:
            x_lag = x[x.index + pd.Timedelta(lag)]
            title = tag + ' [' + lag + '], MI = %.3f' % row.mi
        else:
            x_lag = x
            title = tag + ', MI = %.3f' % row.mi
        y = data[target]
        ax[i].scatter(x_lag.values, y.values, s=3, c=color, cmap='jet', alpha=0.5)
        ax[i].set_title(title, fontsize=9)

    plt.suptitle('%d samples, color shows originating file' % len(data.index))
    plt.savefig(os.path.join(output_dir, 'scatter_top_mi.png'), dpi=200)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs adaptive feature selection by using mutual information.')
    parser.add_argument('config_file', help='YAML config file')
    args = parser.parse_args()
    main(args.config_file)
