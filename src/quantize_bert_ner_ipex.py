# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

"""
 Run the file to quantize the model .
"""

# pylint: disable=E0401, C0413, E0401
# flake8: noqa = E501

import argparse
import pathlib
import random
import warnings
warnings.filterwarnings("ignore")
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.quantization import prepare, convert
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification
)

def main(flags):
    """Run quantization and save the model using IPEX.
    Args:
        flags : run flags
    """
    # Load pretrained tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(flags.model)
    model = AutoModelForTokenClassification.from_pretrained(flags.model)
    model.eval()

    sample_inputs = [
        random.sample(
            range(tokenizer.vocab_size),
            model.config.max_position_embeddings)]
    sample_inputs = tokenizer.batch_decode(sample_inputs)
    sample_inputs = tokenizer(
        sample_inputs[0],
        padding=True,
        max_length=model.config.max_position_embeddings,
        truncation=True
    )

    # INT-8 Dynamic Quantization of Model using IPEX
    qconfig = ipex.quantization.default_dynamic_qconfig
    prepared_model = prepare(
        model, qconfig,
        example_inputs=(
            sample_inputs['input_ids'],
            sample_inputs['attention_mask']),
        inplace=False)

    converted_model = convert(prepared_model)

    path = pathlib.Path(flags.save_model_dir)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        converted_model,
        path / "saved_model_bert_base_ner.pt")
    tokenizer.save_pretrained(path)
    print("Model has been quantized .... & saved in pt format")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--model',
                        type=str,
                        required=True,
                        help="Pretrained BERT based NER model to use"
                        )

    parser.add_argument('--save_model_dir',
                        type=str,
                        required=True,
                        help="directory to save the quantized model to")
    flags = parser.parse_args()
    main(flags)
    