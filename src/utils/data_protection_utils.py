# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 PII data utils .
"""

# pylint: disable=C0325, W0311, C0410, W0622, W0707, C0415, E0401, W1514, W0105, W0238,  W0702
# flake8: noqa = E501

import os, re, json
import string
import pickle
import hashlib
import numpy as np
import numba
import torch
from cryptography.fernet import Fernet
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from utils.timer import AvgCodeTimer

class PII_Data_Detagger():
    '''
    Contains the core anonymization functions
    Also contains the supporting utiliy functions
    '''

    def __init__(self, FLAGS):
        self.format = None
        self.clist = ['a', 'b', 'c', 'd', 'e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
        self.anlist = ['a', 'b', 'c', 'd', 'e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','0','1','2','3','4','5','6','7','8','9']
        self.num = re.compile(r"[0-9]")
        self.alpha = re.compile(r"[a-z]")
        self.alphanum = re.compile(r"[a-z0-9]")
        self.name_generator_model_pkl = './model/name_generator.pkl'
        self.parameters = None
        self.name_seed = 0
        self.domains = "outlook.com"
        self.__load_name_generator()
        self.__load_tokenizer(FLAGS.bert_ner_model)
        self.__load_ipex(FLAGS.ipex)
        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer, aggregation_strategy="max")
        self.avgtimer = AvgCodeTimer('NER Model Inference AverageTime')
        self.avgtimer.ResetTime()

    def reset_avgtimer(self):
        self.avgtimer.ResetTime()

    def avgtime_now(self):
        self.avgtimer.AverageTime()
    
    def __set_format(self, format):
        self.format = format

    def __generate_number(self):
        return self.num.sub(lambda x: str(np.random.randint(0,9)), self.format)


    def __generate_string(self):
        return self.alpha.sub(lambda x: str(np.random.choice(self.clist)), self.format)


    def __generate_alphanum(self):
        return self.alphanum.sub(lambda x: str(np.random.choice(self.anlist)), self.format)


    def __softmax(self,x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)


    def __load_name_generator(self):
        try:
            with open(self.name_generator_model_pkl, 'rb') as f:
                    self.parameters = pickle.load(f)
        except:
            raise Exception("Name generator model loading failed")
        self.ix_to_char = self.parameters['ix_to_char']
        self.char_to_ix = self.parameters['char_to_ix']


    def __load_tokenizer(self, bert_ner_model):
        # Loading the Tokenizer from HF
        self.tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        if os.path.isdir(bert_ner_model) or os.path.isfile(bert_ner_model):
            self.model = torch.load(bert_ner_model)
        else:
            self.model = AutoModelForTokenClassification.from_pretrained(bert_ner_model)


    def __load_ipex(self, ipex):
        if ipex == 1:
            print("Using 'intel_extension_for_pytorch'")
            import intel_extension_for_pytorch as ipex
            self.model = ipex.optimize(self.model)


    def __get_sample_from_index(self, sample_ix, ix_to_char):
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        txt = txt[0].upper() + txt[1:]  # capitalize first character 
        return txt.strip()

    def __get_sample_name(self, seed):
        '''
        Sample a sequence of characters according to a sequence of probability distributions output of the RNN

        Arguments:
        parameters -- python dictionary containing the parameters Waa, Wax, Wya, by, and b. 
        char_to_ix -- python dictionary mapping each character to an index.
        seed -- used for grading purposes. Do not worry about it.

        Returns:
        indices -- a list of length n containing the indices of the sampled characters.
        '''
        parameters = self.parameters
        char_to_ix = self.char_to_ix

        # Retrieve parameters and relevant shapes from "parameters" dictionary
        Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
        vocab_size = by.shape[0]
        n_a = Waa.shape[1]
        
        # Create the one-hot vector x for the first character (initializing the sequence generation).
        x = np.zeros((vocab_size, 1))
        # Initialize a_prev as zeros
        a_prev = np.zeros((n_a, 1))
        
        # Create an empty list of indices, this is the list which will contain the list of indices of the characters to generate
        indices = []
        
        # Idx is a flag to detect a newline character, we initialize it to -1
        idx = -1 
        
        # Loop over time-steps t. At each time-step, sample a character from a probability distribution and append 
        # its index to "indices". We'll stop if we reach 50 characters (which should be very unlikely with a well 
        # trained model), which helps debugging and prevents entering an infinite loop. 
        counter = 0
        newline_character = char_to_ix['\n']
        
        while (idx != newline_character and counter != 50):
            
            # Forward propagate x using the equations (1), (2) and (3)
            a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)
            z = np.dot(Wya, a) + by
            y = self.__softmax(z)
            
            # for grading purposes
            np.random.seed(counter + seed) 
            
            # Sample the index of a character within the vocabulary from the probability distribution y
            idx = np.random.choice(list(range(vocab_size)), p=y.ravel())

            # Append the index to "indices"
            indices.append(idx)
            
            # Overwrite the input character as the one corresponding to the sampled index.
            x = np.zeros((vocab_size, 1))
            x[idx] = 1
            
            # Update "a_prev" to be "a"
            a_prev = a
            
            # for grading purposes
            seed += 1
            counter +=1

        if (counter == 50):
            indices.append(char_to_ix['\n'])
        
        return self.__get_sample_from_index(indices, self.ix_to_char)

      
    def __replace_between(self, text, begin, end, alternative=''):
        '''
        Sample sequence of replacing name
        '''
        middle = text[begin:end]
        return text.replace(middle, alternative, 1) 


    def generate_name(self,val):
        '''
        Sample sequence of generate name 
        '''
        self.name_seed = np.random.randint(0, 4026531839, dtype=np.int64)
        return self.__get_sample_name(self.name_seed)
    

    def obfuscate_number(self, val, nlen):
        '''
        Sample sequence of generate number 
        '''
        format = "9" * nlen
        self.__set_format(format)
        return self.__generate_number()


    def obfuscate_number_format(self,val,format,slen):
        '''
        Sample sequence of generate number to a particular format 
        '''
        self.__set_format(format)
        return self.__generate_number()


    def obfuscate_string_old(self, val, slen):
        '''
        Sample sequence of generate number to a particular format 
        '''
        len = np.random.randint(2,slen)
        format = "x" * len
        self.__set_format(format)
        return  self.__generate_string().title()


    def obfuscate_string(self, val, slen):
        '''
        Sample sequence of generate number to a particular format 
        '''
        return ''.join(np.random.choice(list(string.ascii_letters+"  "), slen)).title()
    
       
    def obfuscate_alphanum(self, slen):
        '''
        Sample sequence of generate alpha numeric         
        '''
        len = np.random.randint(2,slen)
        format = "x" * len
        self.__set_format(format)
        return  self.__generate_alphanum()

        
    def obfuscate_ipv4_address(self,val):
        '''
        Sample sequence of generate ip address          
        '''
        self.__set_format("10.12.999.999")
        return  self.__generate_number()

    def md5hash(self, val, data):
        '''
        Sample sequence of generate utf-8         
        '''
        data = str(data).encode('utf-8')
        return hashlib.md5(data).hexdigest()

    def generate_random_emails(self,val):
        '''
        Sample sequence of generate utf-8         
        '''
        self.name_seed = np.random.randint(0, 4026531839, dtype=np.int64)
        return (str(self.__get_sample_name(self.name_seed))+"@"+str((self.domains)))


    def mask_with_char(self, val, start, end, char):
        '''
        Replace substring with a mask character
        '''
        val = str(val)
        if end==0 or end>len(val): 
            end = len(val)
        return val.replace(val[start:end], (char*(end-start)))


    def encrypt(self, val, key):
        '''
        Encrypt data using the Fernet generated key
        '''
        fernet = Fernet(key)
        val = fernet.encrypt(str(val).encode('utf-8')).decode('utf-8')
        return val


    def detag_pii_data(self, input_text):
        '''
        Identify entities in free flowing text using NER model
        Mask entities with random string
        '''
        with self.avgtimer:
            ner_results = self.nlp(input_text)

        for entity in ner_results:
            #import pdb;pdb.set_trace()
            if entity["entity_group"] in ["PER", "ORG", "LOC"]:
                str_len = (entity["end"] - entity["start"])
                random_string = 'X' * str_len
                input_text = input_text[:entity["start"]] + random_string + input_text[entity["end"]:]
        return input_text


    def generate_store_encyption_key(self, col, key_path, file_path):
        '''
        Generates Fernet keys and stores them in JSON file for decryption
        '''
        key = Fernet.generate_key()
        key_dict = { "keys" : [] }

        try:
            # Check if file exists
            if os.path.isfile(key_path):
                print("## LOG  | Encryption Key file already exists. Appending.")
                with open(key_path,"r") as f:
                    key_dict = json.load(f)
            if "keys" not in key_dict.keys():
                key_dict["keys"] = []
        except:
            key_dict = { "keys" : [] }


        '''
        # Flag to check if the key-value pair exists
        exists = False
        # Iterate through the array of dictionaries
        for i, d in enumerate(key_dict["keys"]):
            # Check if the key-value pair exists
            if d.get("column") == col:
                exists = True
                key_dict[i] = {"column": col,"key": key.decode()}
                break
        if not exists: key_dict["keys"].append({"column": col,"key": key.decode()})
        '''
        key_dict["keys"].append({"column": col,"key": key.decode()})
        key_dict["encrypted_CSV"] = file_path

        # Save modified JSON
        with open(key_path,"w") as f:
                json.dump(key_dict, f, indent=4)

        return key


    '''
    NUMBA optimized functions
    '''
    # @numba.vectorize
    @numba.jit(forceobj=True)
    def data_length(self, val):
        return len(val)

    @numba.jit(forceobj=True)
    def col_mean_length(self, col):
        total_len = 0
        count = 0
        for i in range(col.shape[0]):
            total_len += len(str(col[i]))
            count += 1
        return int(total_len / count)

    @numba.jit(forceobj=True)
    def col_max_length(self, col):
        max_len = 0
        for val in col:
            curr_len =len(val)
            if curr_len > max_len:
                max_len = curr_len
        return max_len
