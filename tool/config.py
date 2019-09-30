#coding=utf-8

import os
import sys
import argparse
import configparser

class MyConf(configparser.ConfigParser):
    def __init__(self, defaults=None):
        configparser.ConfigParser.__init__(self, defaults=defaults)
 
    def optionxform(self, optionstr):
        return optionstr

def loadConfig(config_file):
    config = MyConf()
    config.read(config_file)
    parser = argparse.ArgumentParser()
    for config_key, config_value in config['int'].items():
        parser.add_argument('--%s'%config_key, type=int, default=config.get('int', config_key))
    for config_key, config_value in config['float'].items():
        parser.add_argument('--%s'%config_key, type=float, default=config.get('float', config_key))
    for config_key, config_value in config['str'].items():
        parser.add_argument('--%s'%config_key, type=str, default=config.get('str', config_key))
    args = parser.parse_args()
    return args
