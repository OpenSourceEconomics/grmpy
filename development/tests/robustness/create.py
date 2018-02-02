#!/usr/bin/env python
''' This module creates a mock dataset, which resembles the true one quite
    closely. I merge the local information at random to the baseline
    dataset. The two source files are provided as part of the 
    Carneiro&al.(2011) recomputation material on the AER website.
'''

# standard library
import pandas as pd
import sys
import os

# Set working directory.
dir_ = os.path.abspath(os.path.split(sys.argv[0])[0])
os.chdir(dir_)

''' Read baseline datasets material and merge at random.
'''
basic = pd.read_stata('basicvariables.dta')

local = pd.read_stata('localvariables.dta') 

df    = pd.concat([basic, local], axis = 1)

''' Rename const to _const to avoid error message.
'''
df['_const']  = df['const']

del df['const']


''' Delete redundant columns.
'''
for key_ in ['newid', 'caseid']:
        
    del df[key_]


''' Add squared terms.
'''
for key_ in ['mhgc', 'cafqt', 'avurate', 'numsibs', 'lavlocwage17']:
    
    str_ = key_ + 'sq'
    
    df[str_] = df[key_]**2


''' Store dataset.
'''
df.to_stata('aer-replication-mock.dta')
