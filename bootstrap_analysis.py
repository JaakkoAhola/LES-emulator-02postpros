#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 11:47:25 2021

@author: Jaakko Ahola, Finnish Meteorological Institute
@licence: MIT licence Copyright
"""
print(__doc__)
import time
import pandas

def main():

    ds = {}
    ds["sbday"] = pandas.read_csv("/home/aholaj/mounttauskansiot/puhtiwork/EmulatorManuscriptData/LVL3Day/LVL3Day_bootstrap.csv", index_col = 0)
    ds["sbnight"] = pandas.read_csv("/home/aholaj/mounttauskansiot/puhtiwork/EmulatorManuscriptData/LVL3Night/LVL3Night_bootstrap.csv", index_col = 0)
    ds["salsanight"] = pandas.read_csv("/home/aholaj/mounttauskansiot/puhtiwork/EmulatorManuscriptData/LVL4Night/LVL4Night_bootstrap.csv", index_col = 0)
    ds["salsaday"] = pandas.read_csv("/home/aholaj/mounttauskansiot/puhtiwork/EmulatorManuscriptData/LVL4Day/LVL4Day_bootstrap.csv", index_col = 0)
   
    
    for key in ds:
        # print(ds[key].columns)
        print(f'{key} {ds[key]["RSquared"].mean():.2f} {ds[key]["RSquared"].var():.2f} RMSE: {ds[key]["RMSE"].mean():.3f}')
        
    
if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"\nScript completed in { end - start : .1f} seconds")
