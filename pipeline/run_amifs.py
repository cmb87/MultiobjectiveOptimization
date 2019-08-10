import argparse
import os
from time import process_time

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from amifs import amifs, pre_calc_entropy
from util import get_logger, get_starts, get_tags


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
    entropies = pre_calc_entropy(data, tags)
    logger.info('Entropy calculations took %.1f s' % (process_time() - t0))

    mi = {}
    targets = config['targets']
    for target in targets:
        logger.info('Target = ' + target)
        inputs = set(tags.copy())
        inputs.remove(target)

        if not config['allow_targets_as_inputs']:
            inputs -= set(targets)
            logger.info('Ignoring any other target as input.')

        t0 = process_time()
        fs, ic, mi, fi_score = amifs(data, inputs, target, config['max_features'], Hs=entropies, MIs=mi)
        logger.info('amifs took %.1f s' % (process_time() - t0))

        # Adds tags descriptions
        description = []
        for tag in fs:
            try:
                description.append(tag_list.loc[tag].item())
            except KeyError:
                description.append('')

        df = pd.DataFrame({'tag': fs,
                           'MI(%s)' % target: [ic[f] for f in fs],
                           'description': description})
        df.to_csv(output_dir + '/%s__amifs_maxfeatures_%d.csv' % (target, len(fs)))

        # Color-code the origin (file) of each sample, mapping each string to an int
        color = [datasets.index(s) for s in data['dataset']]
        fig, ax = plt.subplots(3, 5, figsize=(25, 15), sharey=True)
        ax = ax.ravel()
        y = data[target]
        for i in range(min(15, len(fs))):
            tag = fs[i]
            x = data[tag]
            title = tag + ', MI = %.3f' % ic[tag]
            ax[i].scatter(x.values, y.values, s=3, c=color, cmap='jet', alpha=0.5)
            ax[i].set_title(title, fontsize=9)
            ax[i].set_xlabel(tag, fontsize=9)
        for i in range(0, 15, 5):
            ax[i].set_ylabel(target, fontsize=9)

        plt.suptitle('%s (%d samples)' % (str(datasets), len(data.index)))
        plt.savefig(output_dir + '/%s__scatter_top_amifs.png' % target, dpi=200)

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs adaptive feature selection by using mutual information.')
    parser.add_argument('config_file', help='YAML config file')
    args = parser.parse_args()
    main(args.config_file)
