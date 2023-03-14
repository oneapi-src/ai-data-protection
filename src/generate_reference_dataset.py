# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

"""
 Data generation  file .
"""

# pylint: disable=C3001, C0410, C0321, W0611
# flake8: noqa = E501

import random,string,argparse
import pandas as pd
import numpy as np


def generate_arr(N, text_array):
    '''
    List of randomization options for each category ( e.g. name, phone ,etc) are provide
    Generates an array of the random generated strings
    The order and selection of the categories can be customized
    The same should be updated in the columns array too
    '''
    len_text_array = len(text_array)

    name = lambda : ''.join(random.choices(string.ascii_lowercase + "   ", k=10)).title()  # Name
    phone = lambda : "+" + "".join(random.choices(string.digits, k=2)) + " " +  ''.join(random.choices(string.digits, k=10))  # Phone
    location = lambda : ''.join(random.choices(string.ascii_lowercase + " ", k=20)).title()  # Location
    email = lambda : ''.join(random.choices(string.ascii_lowercase, k=10))+"@"+''.join(random.choices(string.ascii_lowercase + "", k=5))+".com"  # Email
    passport = lambda : ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))  # Passport
    ipv4 = lambda : ".".join(str(random.randint(0,255)) for _ in range(4))  # IP address
    invoice = lambda : ''.join(random.choices(string.ascii_uppercase + string.digits + string.ascii_lowercase, k=15))  # Invoice
    credit_card = lambda : random.randint(10**16-1,10**17-1)  # Credit Card
    ssn = lambda : random.randint(10**9-1,10**10-1)  # Social Security
    text = lambda : text_array[random.randint(0,len_text_array-1)] # Text data from available sources

    arr = [ [name(),phone(),location(),email(),passport(),ipv4(),invoice(),credit_card(),credit_card(),ssn(),text() ] for i in range(N) ]
    columns = ["Name","phone_number","location","Email","passport_number","ip_address","invoice_number","credit_card_number","debit_card_number","social_security_number","description"]

    return arr, columns


def load_wikiData(wiki_path):
    '''
    Loads the People Wiki Data
    Link - https://www.kaggle.com/datasets/sameersmahajan/people-wikipedia-data?resource=download
    Text column is exported as a 1-D Array
    '''
    wiki_df = pd.read_csv(wiki_path)
    wiki_df = wiki_df["text"].apply(str.title)
    return wiki_df.to_numpy()


def generate_CSV(N, wiki_path, out_path=None):
    '''
    Creates a dataframe using the 10 standard columns provided in the sample configuration JSON
    Also, adds a text column using descriptions from the People Wiki Dataset - useful for NER task tests
    The dataframe is then saved as a CSV, which can be used for anonymization verification tasks
    '''
    text_array = load_wikiData(wiki_path)
    arr,columns = generate_arr(N,text_array)
    
    # Create Dataframe
    df = pd.DataFrame(arr,columns=columns)

    # Save dataframe to CSV
    if out_path is None: out_path = "../data/demo_pii.csv"
    df.to_csv(out_path,index=False)




if __name__ == "__main__":
    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-n',
                        '--number',
                        type=int,
                        required=False,
                        default=1000,
                        help='Provide the number of records to generate') 
    parser.add_argument('-d',
                        '--data',
                        type=str,
                        required=True,
                        help='Provide the path to People Wiki Dataset')
    parser.add_argument('-o',
                        '--output',
                        type=str,
                        required=True,
                        help='Provide the path to save the reference dataset')
    args = parser.parse_args()
    total_records = args.number
    wiki_path = args.data
    out_path = args.output

    print(f"Generating Random Dataset in {out_path}...")
    generate_CSV(total_records, wiki_path, out_path)
    print("DONE")
