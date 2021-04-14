#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 17:16:48 2021

@author: Jaakko Ahola, Finnish Meteorological Institute
@licence: MIT licence Copyright
"""
print(__doc__)
import time
import pandas
from sklearn.metrics import mean_squared_error

from PostProcessingMetaData import PostProcessingMetaData





def main():
# (self, name : str, trainingSimulationRootFolder : list, dataOutputRootFolder : list, configFile = None, figureFolder = None)
    meta = PostProcessingMetaData("sopo",
                                  ["/home/aholaj/mounttauskansiot/puhtiwork/eclair_training_simulations/"], ["/home/aholaj/mounttauskansiot/puhtiwork/EmulatorManuscriptData/"],
                                  "/home/aholaj/mounttauskansiot/puhtiwork/EmulatorManuscriptData/phase02.yaml")

    ds = {}
    ds["SB Day"] = pandas.read_csv("/home/aholaj/mounttauskansiot/puhtiwork/EmulatorManuscriptData/LVL3Day/LVL3Day_complete.csv", index_col = 0)
    ds["SB Night"] = pandas.read_csv("/home/aholaj/mounttauskansiot/puhtiwork/EmulatorManuscriptData/LVL3Night/LVL3Night_complete.csv", index_col = 0)
    ds["SALSA Night"] = pandas.read_csv("/home/aholaj/mounttauskansiot/puhtiwork/EmulatorManuscriptData/LVL4Night/LVL4Night_complete.csv", index_col = 0)
    ds["SALSA Day"] = pandas.read_csv("/home/aholaj/mounttauskansiot/puhtiwork/EmulatorManuscriptData/LVL4Day/LVL4Day_complete.csv", index_col = 0)
    
    
    # rms = mean_squared_error(y_actual, y_predicted, squared=False)
    errors = {}
    for key in ds:
        errors[key] = {}
        
        alijoukko = ds[ key ][ ds[key][meta.filterIndex] ]
        
        errors[key]["emul"]      = mean_squared_error( alijoukko[ meta.responseVariable ], alijoukko[ meta.emulatedVariable ], squared=False )
        errors[key]["linear"]    = mean_squared_error( alijoukko[ meta.responseVariable ], alijoukko[ meta.linearFitVariable ], squared=False )
        errors[key]["corrected"] = mean_squared_error( alijoukko[ meta.responseVariable ], alijoukko[ meta.correctedLinearFitVariable ], squared=False )
        
        print(f'{key} & {errors[key]["emul"]:.4f} & {errors[key]["linear"]:.4f} & {errors[key]["corrected"]:.4f} \\')
        print("\hline")
if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"\nScript completed in { end - start : .1f} seconds")
