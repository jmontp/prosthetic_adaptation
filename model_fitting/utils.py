#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  8 13:17:39 2021

@author: jmontp
"""

import numpy as np

test_assertions = True
def assert_pd(matrix,name):
    if test_assertions:
        try: 
            assert (matrix.shape[0] == matrix.shape[1])
            assert (len(matrix.shape)==2)
        except AssertionError:
            print(name + " NOT EVEN SQUARE: " + str(matrix.shape))
            print("Assertion on matrix: \n{}".format(matrix))
    
            raise AssertionError
    
        try:
            assert (np.linalg.norm(matrix-matrix.T) < 1e-7*np.linalg.norm(matrix))
        except AssertionError:
            print(name + " Error with norm: " + str(np.linalg.norm(matrix-matrix.T)))
            print("Assertion on matrix: \n{}".format(matrix))
    
            raise AssertionError
            
        try:
            for e in np.linalg.eigh(matrix)[0]:
                assert (e + 1e-8 > 0)
        except AssertionError:
            print(name + " Error with Evalue: " + str([e for e in np.linalg.eigh(matrix)[0]]))
            print("Assertion on matrix: \n{}".format(matrix))
            raise AssertionError
        else:
            return None    