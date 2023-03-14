# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

"""
 Timer file .
"""

# pylint: disable=R0903, C0115
# flake8: noqa = E501

import time

class FunctionTimer:
    """
    Timer class that can be used as a decorator to time the execution of a function.
    """
    def __init__(self):
        self.start_time = 0
        self.end_time = 0
    
    def __call__(self, func):
        """
        The __call__ method allows the class to be used as a decorator.
        """
        def wrapper(*args, **kwargs):
            """
            The wrapper function is used to time the execution of the decorated function.
            """
            self.start_time = time.time()
            result = func(*args, **kwargs)
            self.end_time = time.time()
            print(f'## TIME | {func.__name__} took {self.end_time - self.start_time:.6f}s')
            return result
        return wrapper


class CodeTimer:
    def __init__(self, name=None):
        self.name = " '"  + name + "'" if name else ''
        self.start_time = 0
        self.end_time = 0

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        print(f'## TIME | {self.name} took {self.end_time - self.start_time:.6f}s')


class AvgCodeTimer:
    def __init__(self, name=None):
        self.name = " '"  + name + "'" if name else ''
        self.start_time = 0
        self.end_time = 0
        self.totaltime = 0
        self.iterations = 0

    def __enter__(self):
        self.start_time = time.time()
        self.end_time = 0

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        self.totaltime = self.totaltime + (self.end_time - self.start_time)
        self.iterations = self.iterations + 1
        # print(f'## TIME | {self.name} took {self.end_time - self.start_time:.6f}s')

    def AverageTime(self):
        avgtime = self.totaltime / self.iterations
        print(f'## TIME | {self.name} took Avg - {avgtime:.6f}s and Total - {self.totaltime:.6f}s')

    def ResetTime(self):
        self.start_time = 0
        self.end_time = 0
        self.totaltime = 0
        self.iterations = 0
