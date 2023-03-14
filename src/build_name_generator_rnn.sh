#!/bin/bash

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# flake8: noqa = E501

echo "checking model directory path to save the model"
DIR="model"
if [ -d "$DIR" ]; 
then
    echo "Directory exist ,script is getting executed ..."
else
    mkdir ./model
    echo "File created"
fi
## Clone the Name Generator Repository
cd src
echo "> Cloning the Repository..."
git clone https://github.com/chaitanyarahalkar/Name-Generator-RNN
cd Name-Generator-RNN/

## Convert the notebook to executable python script
echo "> Converting the notebook to executable python script..."
pip install  ipynb-py-convert
ipynb-py-convert ./Name-Generator.ipynb ./build_name_generator_rnn.py

## Apply the patch to save the model as pickle file
echo "> Applying the patch required for saving the RNN model..."
patch -p1 ./build_name_generator_rnn.py < ../build_name_generator_rnn.patch

## Execute the module to create the pickle file
echo "> Execute the python script to create the RNN model..."
python ./build_name_generator_rnn.py

## Copy the pickle file
echo "> Copying the model file..."
cp ./name_generator.pkl ../../model

cd ../
echo "> Script Execution COMPLETED"

