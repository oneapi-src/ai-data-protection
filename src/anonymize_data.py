# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run the file to protect PII data  .
"""

# pylint: disable=C0410, R1711, W1514,  E0401,  C0412, R1732
# flake8: noqa = E501

import os,argparse, json
import sys
from pathlib import Path
from utils.data_protection import DataProtection

# Utility to measure execution time for indvidual functions
from utils.timer import FunctionTimer


@FunctionTimer()
def execute_all_operations(df, config, FLAGS):
    '''
    Iterates through each operation listed in the JSON configuration file
    Executes relevant function based on the `type` of the operation

    Arguments:
        df (pandas.DataFrame): The dataframe to be masked
        config (str): Configuration dict, containing operations to be performed
        FLAGS (dict): Parameters for execution mode
    '''
    dataProtection = DataProtection(df, FLAGS)
    print('-'*20)

    for operation in config["operation"]:

        print(f'Anonymizing the column \'{operation["column"]}\' for the type \'{operation["type"]}\' ...')
        
        if operation["type"] == "string":
            dataProtection.mask_string(operation)
       
        if operation["type"] == "numeric":
            dataProtection.mask_numeric(operation)

        if operation["type"] == "name":
            dataProtection.mask_name(operation)

        if operation["type"] == "phone":
            dataProtection.mask_phone(operation)

        if operation["type"] == "ip_address":
            dataProtection.mask_ip(operation)
                 
        if operation["type"] == "email_address":
            dataProtection.mask_email(operation)

        if operation["type"] == "alphanumeric":
            dataProtection.mask_invoice_number(operation)
       
        if operation["type"] == "mask":
            dataProtection.mask_with_char(operation)

        if operation["type"] == "hash":
            dataProtection.mask_ssn(operation)
    
        if operation["type"] == "encrypt":
            dataProtection.encrypt(operation, config["output_file"], config["encryption_key_file"])
   
        if operation["type"] == "text":
            dataProtection.obfuscate(operation)
     
    return df


@FunctionTimer()
def read_input_dataset(csvfile):
    return pd.read_csv(csvfile)


@FunctionTimer()
def write_output_dataset(dataframe):
    dataframe.to_csv(config['output_file'],index=False)
    return None


def read_files(json_path):
    '''
    Reads the config JSON file from the path
    And the corresoponding CSV mentioned in the config
    '''
    # Load JSON
    with open(json_path,'r') as f:
        config = json.load(f)

    # Loading input CSV
    df= read_input_dataset(config['input_file'])
    

    return config, df




if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        required=False,
                        default="./config/reference_dataset_config.json",
                        help='Provide the configuration JSON file. Default - `./config/reference_dataset_config.json`')
    parser.add_argument('-b',
                        '--bert_ner_model',
                        type=str,
                        required=False,
                        default="dslim/bert-base-NER",
                        help='Provide the BERT based NER model file, from hugging-face repository or saved model file. Default - `dslim/bert-base-NER`')
    parser.add_argument('-i',
                        '--ipex',
                        action='store_true',
                        help='Use IPEX optimized inference.')
    parser.add_argument('-m',
                        '--modin',
                        action='store_true',
                        help='Use Modin instead of Pandas. Useful while dealing with large datasets.')

    parser.add_argument('-l',
                        '--logfile',
                        type=str,
                        default="",
                        help="log file to output benchmarking results to")
    parser.add_argument('-v',
                        '--verbose',
                        action='store_true',
                        help='Show progress bars during execution.')
    FLAGS = parser.parse_args()

    # Get path of the config file
    json_path=FLAGS.config

    # Select pandas/modin
    if FLAGS.logfile == "":
        print("As output file name has not provided log would not get printed")
    else: 
        print("Log will get printed to the mentioned logfile ,please wait ..")
        os.makedirs('./log', exist_ok=True)
        old_stdout = sys.stdout
        log_file = open('./log/'+ FLAGS.logfile+".log","w")
        sys.stdout = log_file

    if FLAGS.modin:
        print("Using Modin")
        if FLAGS.verbose:
            FLAGS.verbose = False
            print("Verbose option is not allowed with Modin, progress bars will be disabled")
        import modin.config as cfg  # pylint: disable=E0401
        cfg.Engine.put('ray')
        import ray  # pylint: disable=E0401
        ray.init()
        RAY_DISABLE_MEMORY_MONITOR = 1
        import modin.pandas as pd
        pd.Series.progress_apply = pd.Series.apply
    else:
        print("Using Pandas")
        import pandas as pd
        pd.Series.progress_apply = pd.Series.apply

    # Read config and dataset files
    print('Reading input dataset')
    config, df = read_files(json_path)
    
    #validating keys
    config_encrypted = open(FLAGS.config)
    key_stored = json.load(config_encrypted)
    print("These is to validate right encrypted keys")
    keys_read=key_stored['encryption_key_file'] 
    if os.path.isfile(keys_read):
        os.remove(keys_read)

    # Start operations to get anonymize the CSV file
    output_df = execute_all_operations(df, config, FLAGS)
    write_output_dataset(output_df)
    print(f"Anonymization Complete. The output is saved here - {config['output_file']}")
