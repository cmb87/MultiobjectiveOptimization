import os
import sys
from glob import glob
from time import process_time

import numpy as np
import pandas as pd
import yaml
import logging

"""
Creating HDF5 file with content of all CSV files.
Entries with "ERROR" are replaced with NaN.

Limitation: columns with non-float values (e.g. strings) are currently ignored.

"""


def get_logger(log_file):
    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def get_dataset_name(filename):
    return os.path.basename(filename).replace('.csv.zip', '').replace('.', '_')


def import_csv(csv_file, logger):
    t0 = process_time()

    # BDAS encodes sensor failures as -9999
    df = pd.read_csv(csv_file, parse_dates=['Time'], na_values=['ERROR', '-9999', '-9999.0'])
    df.set_index(['Time'], inplace=True, verify_integrity=True)
    for col in df.columns:
        if not df[col].dtype is np.dtype(np.float):
            # Try to convert the individual elements of the bad column to float.
            val = np.empty((len(df.index), ))
            for i, x in enumerate(df[col].values):
                try:
                    val[i] = np.float(x)
                except ValueError:
                    val[i] = np.NaN
                    logger.warning(col + ': cannot parse "%s" as float, replaced by NaN.' % str(x))
            df[col] = pd.Series(val, index=df.index)
    logger.info('%s: read_csv took %.2fs' % (os.path.basename(csv_file), process_time() - t0))
    return df


def main(config_file):

    with open(config_file) as yaml_config_file:
        config = yaml.load(yaml_config_file, Loader=yaml.FullLoader)
        input_files = config['input_files']
        h5_output_file = config['h5_output_file']
        logger = get_logger(config['log_file'])

    files = glob(input_files)
    logger.info('######## Logger started with ' + config_file)
    logger.info('Found %d files using given file name pattern.' % len(files))

    with pd.HDFStore(h5_output_file, mode='a', complevel=9) as store:
        for file in files:
            df = import_csv(file, logger)
            name = get_dataset_name(file)
            store.put(name, df)


if __name__ == '__main__':
    main(sys.argv[1])
