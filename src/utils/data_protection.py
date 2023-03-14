# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 PII data protection .
"""

# pylint: disable=C0321, W0611, C0411, C0412
# flake8: noqa = E501

import time
import numpy as np
from cryptography.fernet import Fernet
from utils.data_protection_utils import PII_Data_Detagger
from utils.config_validation import ValidationTools

# Utility to measure execution time for indvidual functions
from tqdm import tqdm
from utils.timer import CodeTimer


class DataProtection():
    """
    The DataProtection class is responsible for carrying out a series of operations
    These include, calling the validation operation to check the integrity of the data columnd and the operation requested,
        calculating key data column details such as length, mean ,etc
        finally, utilizing the appropriate util function to perform the anonymization task

    Attributes:
        df (pandas.DataFrame): The dataframe to be masked
        FLAGS (dict): Parameters for execution mode

    Attributes:
        utilities: Instance of the PIIDataDetagger class (performs the anonymization util functions)
        validate: Instance of the ValidationTools class (performs validation operations)
    """


    def __init__(self, df, FLAGS) -> None:
        '''
        Initializes the class
        '''
        self.tqdmbar = ' >='
        self.tqdmbarformat = '{l_bar}{bar}| {n_fmt}/{total_fmt}'
        self.df = df
        self.utilities = PII_Data_Detagger(FLAGS)
        self.validate = ValidationTools()
        if FLAGS.verbose: tqdm(position=0, leave=True, bar_format='{desc}: {elapsed} {percentage:3.0f}%|{bar}|{r_bar}').pandas(desc='Progress',ascii=self.tqdmbar,bar_format=self.tqdmbarformat)

    def mask_string(self,operation) -> None:
        self.validate.validate_string(operation)
        length = self.utilities.col_mean_length(self.df[operation['column']].to_numpy())
        with CodeTimer('mask_string'):
            self.df[operation["column"]] = self.df[operation["column"]].progress_apply(self.utilities.obfuscate_string,slen=length)

    def mask_numeric(self,operation) -> None:
        self.validate.validate_numeric(operation)
        length = operation['len'] if operation['len'] else self.utilities.col_mean_length(self.df[operation['column']].to_numpy())
        with CodeTimer('mask_numeric'):
            self.df[operation["column"]] = self.df[operation["column"]].progress_apply(self.utilities.obfuscate_number,nlen=length)
    
    def mask_name(self,operation) -> None:
        self.validate.validate_name(operation)
        with CodeTimer('name_generation'):
            self.df[operation["column"]] = self.df[operation["column"]].progress_apply(self.utilities.generate_name)
    
    def mask_phone(self,operation) -> None:
        self.validate.validate_phone(operation) 
        length = operation['len'] if operation['len'] else self.utilities.col_mean_length(self.df[operation['column']].to_numpy())
        with CodeTimer('mask_numeric_with_prefix'):
            self.df[operation["column"]] = operation['prefix']+"-"+ self.df[operation["column"]].progress_apply(self.utilities.obfuscate_number,nlen=length)
    
    def mask_ip(self,operation) -> None:
        self.validate.validate_ip(operation)
        with CodeTimer('mask_ipv4_address'):
            self.df[operation["column"]] =  self.df[operation["column"]].progress_apply(self.utilities.obfuscate_ipv4_address)
    
    def mask_email(self,operation) -> None:
        self.validate.validate_email(operation)
        with CodeTimer('mask_email_address'):
            self.df[operation["column"]] =  self.df[operation["column"]].progress_apply(self.utilities.generate_random_emails)
    
    def mask_invoice_number(self,operation) -> None:
        self.validate.validate_invoiceno(operation) 
        length = operation['len'] if operation['len'] else self.utilities.col_mean_length(self.df[operation['column']].to_numpy())
        with CodeTimer('mask_formatted_numeric'):
            self.df[operation["column"]] =  self.df[operation["column"]].progress_apply(self.utilities.obfuscate_number_format,format="999-999-999-999-999-99999",slen=length)
    
    def mask_ssn(self,operation) -> None:
        self.validate.validate_ssn(operation)
        with CodeTimer('string_hash'):
            self.df[operation["column"]] =  self.df[operation["column"]].progress_apply(self.utilities.md5hash,data="abcdefghij123456")
    
    def mask_with_char(self,operation) -> None: 
        self.validate.validate_mask_with_char(operation) 
        if "start" not in operation.keys(): start = 0 
        if "end" not in operation.keys(): end = 0 
        if "char" not in operation.keys(): char = "X"
        with CodeTimer('mask_string_partly'):
            self.df[operation["column"]]=self.df[operation["column"]].progress_apply(self.utilities.mask_with_char,start=start,end=end,char=char)   
    
    def encrypt(self,operation, data_file_path, key_file_path) -> None:
        self.validate.validate_encrypt(operation)
        key = self.utilities.generate_store_encyption_key(operation["column"],key_file_path,data_file_path)
        with CodeTimer('encrypt_string'):
            self.df[operation["column"]]=self.df[operation["column"]].progress_apply(self.utilities.encrypt, key=key)
    
    def obfuscate(self,operation) -> None:
        self.validate.validate_obfuscate(operation)
        self.utilities.reset_avgtimer()
        self.df[operation["column"]] =  self.df[operation["column"]].progress_apply(self.utilities.detag_pii_data)
        self.utilities.avgtime_now()
