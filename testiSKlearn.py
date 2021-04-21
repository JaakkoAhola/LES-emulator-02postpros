#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 10:13:40 2021

@author: Jaakko Ahola, Finnish Meteorological Institute
@licence: MIT licence Copyright
"""
print(__doc__)
import numpy
import time
import pandas
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

def main():
    inputvars = ['q_inv',
             'tpot_inv',
             'tpot_pbl',
             'lwp',
             'pblh',
             'cos_mu',
             'cdnc',
             'ks',
             'as',
             'cs',
             'rdry_AS_eff']    

    df = pandas.read_csv("/home/aholaj/mounttauskansiot/puhtiwork/EmulatorManuscriptData/LVL3Night/LVL3Night_complete.csv", index_col = 0)
    dd = df[df['responseVariablew2pos:cfracEndValue>=0.61;cloudTopRelativeChange<=1.1;w2pos!=-999;.']]

    kfold = KFold(n_splits = 10, shuffle = True, random_state = 0)

    inputvars = list(set(dd.keys()).intersection(inputvars))
    outputAccVar = 'w2pos'
    variables = inputvars + [outputAccVar]
    filt = dd[variables]
    filt
    ff = filt

    k = 0
    for trainIndex, testIndex in kfold.split(ff):
    
        if k == 0:
            break
        
    reg = LinearRegression().fit(dd.iloc[trainIndex]["drflx"].values.reshape(-1,1), dd["w2pos"].iloc[trainIndex].values.reshape(-1,1))
    pred = reg.predict(dd.iloc[testIndex]["drflx"].values.reshape(-1,1))
    
    true = dd.iloc[testIndex]["w2pos"].values.reshape(-1,1)
    
    print(numpy.abs(pred-true))
    
    
if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"\nScript completed in { end - start : .1f} seconds")
