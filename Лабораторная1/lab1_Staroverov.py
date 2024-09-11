# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 19:39:09 2024

@author: iliys
"""

import numpy as np

rnd = np.random.default_rng()
a = rnd.random((3, 3))
print('a =\n', a)

a_normilized = (a - np.min(a)) / (np.max(a) - np.min(a))
print('normilzed a =\n', a_normilized)