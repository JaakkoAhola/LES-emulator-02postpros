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
    observationParameters = {}
    observationParameters["slope"] = -0.44/100.
    observationParameters["intercept"] = 22.30/100.
    observationParameters["error"] = 13./100.

    ds = {}
    ds["SB Day"] = pandas.read_csv("/home/aholaj/mounttauskansiot/puhtiwork/EmulatorManuscriptData/LVL3Day/LVL3Day_complete.csv", index_col = 0)
    ds["SB Night"] = pandas.read_csv("/home/aholaj/mounttauskansiot/puhtiwork/EmulatorManuscriptData/LVL3Night/LVL3Night_complete.csv", index_col = 0)
    ds["SALSA Night"] = pandas.read_csv("/home/aholaj/mounttauskansiot/puhtiwork/EmulatorManuscriptData/LVL4Night/LVL4Night_complete.csv", index_col = 0)
    ds["SALSA Day"] = pandas.read_csv("/home/aholaj/mounttauskansiot/puhtiwork/EmulatorManuscriptData/LVL4Day/LVL4Day_complete.csv", index_col = 0)
    
    
    # rms = mean_squared_error(y_actual, y_predicted, squared=False)
    analysis = {}
    for key in ds:
        
        analysis[key] = {}
        
        alijoukko = ds[ key ][ ds[key][meta.filterIndex] ]
        
        analysis[key]["size"] = alijoukko.shape[0]
        
        analysis[key]["outside"]      = alijoukko[ ~ (( alijoukko[meta.responseVariable] > alijoukko["drflx"]*observationParameters["slope"]+ observationParameters["intercept"]-observationParameters["error"]) & \
                                     (alijoukko[meta.responseVariable] < alijoukko["drflx"]*observationParameters["slope"]+ observationParameters["intercept"]+observationParameters["error"]))].shape[0]
        
        analysis[key]["inside"] = analysis[key]["size"] - analysis[key]["outside"]
        
        analysis[key]["percentageInside"] = analysis[key]["inside"] / analysis[key]["size"] * 100.
        
        print(f'{key} & {analysis[key]["size"]} & {analysis[key]["inside"]} & {analysis[key]["percentageInside"]:.1f} \\')
        print("\middlehline")
if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"\nScript completed in { end - start : .1f} seconds")
