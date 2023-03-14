# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

"""
 Run the file for data decryption .
"""

# pylint: disable=W1514
# flake8: noqa = E501

import json
import argparse
import pandas as pd
from cryptography.fernet import Fernet



def read_files(json_path):
    '''
    Reads the config JSON file from the path
    And the corresoponding CSV mentioned in the config
    '''
    # Load JSON
    with open(json_path,'r') as f:
        config = json.load(f)

    # Loading input CSV
    df= pd.read_csv(config['encrypted_CSV'])
    print(df)
    
    return config, df


def decrypt(val,fernet_key):
    '''
    Performs Decryption using the Fernet Key
    Data from the CSV is converted into bytes, and the decrypted value is decoded back to string
    '''
    return fernet_key.decrypt(bytes(val,encoding="utf-8")).decode('utf-8')
     


def execute_decryptions(df, config):
    '''
    Runs the decryption task for each encrypted column
    '''
    for data in config["keys"]:
        print("The encrypted columns and keys are ....", data.values())
        key = data["key"]
        fernet_key = Fernet(key)
        df[data["column"]] = df[data["column"]].apply(decrypt,fernet_key=fernet_key)
    return df


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        required=False,
                        default='./config/keys.json',
                        help='Provide the encryption keys JSON file. Default - `./config.json`')
    parser.add_argument('-o',
                        '--output',
                        type=str,
                        required=False,
                        default="../data/decrypted_file.csv",
                        help='Provide path where the decrypted file should be saved')
    FLAGS = parser.parse_args()
    json_path = FLAGS.config
    out_path = FLAGS.output
    
    # Reads the encrypted file and key details 
    config, df = read_files(json_path)
           
    
    # Start operations to decrypt the CSV file
    output_df = execute_decryptions(df, config)
    output_df.to_csv(out_path, index=False) # updated_dataframe
    df= pd.read_csv(config['encrypted_CSV']) # old_dataframe
    print("Above mentioned encryted columns have been decrypted, printing the updated coulmns value only of the dataframe........." +' \n where [self] represents Encrypted values & [other] reprents new values after decryption')
    # Displaying updated columns values only for encrypted columns 
    print(df.compare(output_df))
    