# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

"""
 config file validation.
"""

# flake8: noqa = E501
# pylint: disable=W0311, C0123


class ValidationTools():
    """
    The ValidationTools class is resposible for carrying out a series of validation operations
    These include:
        1. Checking if column exists
        2. Check if the datatypes are correct
        3. Check if data is valid
        # TODO - add more
    """

    def __init__(self) -> None:
         pass

    def validate_string(self,operation):
        if operation["column"] not in operation.values():
            raise Exception("Key error")
        if "len" in operation.keys():
            if operation["len"] is None or type(operation["len"])!=int:
                raise TypeError("Invalid JSON Syntax, for mask_with_char operation")
        if "len" not in operation.keys():
            operation["len"] = None
        
    def validate_numeric(self,operation):
        if operation["column"] not in operation.values():
            raise Exception("Key error")
        if "len" not in operation.keys():
            operation["len"] = None

    def validate_name(self,operation):
        if operation["column"] not in operation.values():
            raise Exception("Key error")

    def validate_phone(self,operation):
        if operation["column"] not in operation.values():
            raise Exception("Key error")
        if "len" not in operation.keys():
            operation["len"] = None

    def validate_ip(self,operation):
        if operation["column"] not in operation.values():
            raise Exception("Key error")

    def validate_email(self,operation):
        if operation["column"] not in operation.values():
            raise Exception("Key error")

    def validate_invoiceno(self,operation):
        if operation["column"] not in operation.values():
            raise Exception("Key error")
        if "len" not in operation.keys():
            operation["len"] = None

    def validate_ssn(self,operation):
        if operation["column"] not in operation.values():
            raise Exception("Key error")

    def validate_mask_with_char(self,operation):
        if operation["column"] not in operation.values():
            raise Exception("Key error")
        if "start" in operation.keys():
            if operation["start"] is None or type(operation["start"])!=int:
                raise TypeError("Invalid JSON Syntax, for mask_with_char operation")
        if "end" in operation.keys():
            if operation["end"] is None or type(operation["end"])!=int:
                raise TypeError("Invalid JSON Syntax, for mask_with_char operation")
        if "char" in operation.keys():
            if operation["char"] is None or type(operation["end"])!=str:
                raise TypeError("Invalid JSON Syntax, for mask_with_char operation")

    def validate_encrypt(self,operation):
        if operation["column"] not in operation.values():
            raise Exception("Key error")

    def validate_obfuscate(self,operation):
        if operation["column"] not in operation.values():
            raise Exception("Key error")
            