import os
import logging
import configparser
import random

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def make_train_subset(config):
    data_path = os.path.join(*config['data_path']['train_dataset'].split('\\'))    # make sure the path works on both windows and linux
    with open(data_path, 'r') as f:
        data = f.readlines()
        
    train_subset = data[:500]
    train_subset_path = os.path.join(*config['data_path']['train_subset'].split('\\'))
    write_to_file(train_subset, train_subset_path)
    
    
def split_train_test(data, train_size):
    random.shuffle(data)
    train_data = data[:int((train_size)*len(data))]
    test_data = data[int((train_size)*len(data)):]
    
    return train_data, test_data

def write_to_file(data, data_path):
    with open(data_path, 'w') as f:
        for sample in data:
            f.write(sample)
    
    logger.info(f'Write dataset to {data_path}')

def train_val_test_split(config):
    data_path = os.path.join(*config['data_path']['raw_dataset'].split('\\'))    # make sure the path works on both windows and linux
    with open(data_path, 'r') as f:
        data = f.readlines()
    
    train_valid_data, test_data = split_train_test(data, float(config['data']['train_size']))
    train_data, valid_data = split_train_test(train_valid_data, float(config['data']['train_size']))
    
    train_path = os.path.join(*config['data_path']['train_dataset'].split('\\'))
    test_path = os.path.join(*config['data_path']['test_dataset'].split('\\'))
    val_path = os.path.join(*config['data_path']['val_dataset'].split('\\'))
    
    write_to_file(train_data, train_path)
    write_to_file(test_data, test_path)
    write_to_file(valid_data, val_path)

if __name__ == '__main__':
    config_path = os.path.join('configs', 'config.cfg')
    config = configparser.ConfigParser()
    config.read(config_path)
    random.seed(int(config['general']['seed']))
    
    make_train_subset(config)
    
    