#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 14:58:24 2020

@author: Jaakko Ahola, Finnish Meteorological Institute
@licence: MIT licence Copyright
"""
import time
import pandas


    
def merge(dictionary):
    responseFile = dictionary["folder"] + "/" + dictionary["longID"] + "_responseFromTrainingSimulations.csv"
    spdfFile = dictionary["folder"] + "/" + dictionary["longID"] + "_simulatedVSPredictedData.csv"
    
    fitFile = dictionary["folder"] + "/" + dictionary["longID"] + "_fit.csv"
    fit = pandas.read_csv(fitFile)
    
    response = pandas.read_csv(responseFile)
    response["ID"] = response.apply(lambda row: dictionary["ID_prefix"] +"_" + str(int(row["i"])).zfill(3), axis = 1)
    response["wpos_linearFit"] = fit["linearFit"]
    response = response.drop(columns="i")
    
    
    spdf = pandas.read_csv(spdfFile)
    spdf["ID"] = spdf.apply(lambda row: dictionary["ID_prefix"] +"_" + str(int(row["designCase"] + 1)).zfill(3), axis = 1)
    
    merged = pandas.merge(spdf, response, on=["ID"], how = "outer")
    
    merged["wpos"] = merged["wpos_x"]
    merged = merged.drop(columns = ["wpos_x", "wpos_y", "designCase"])
    
    merged.set_index("ID", inplace = True )
    
    merged.to_csv("/home/aholaj/Data/EmulatorManuscriptData/Datasets/"  + dictionary["longID"] +  "_input_output_merged.csv", index_label = "ID")

def main():
        
    listOfDatasets = [{"folder" : "/home/aholaj/Data/EmulatorManuscriptData/LVL3Day/",
    "longID" : "LVL3Day",
    "ID_prefix" : "3D"},
    
    {"folder" : "/home/aholaj/Data/EmulatorManuscriptData/LVL3Night/",
    "longID" : "LVL3Night",
    "ID_prefix" : "3N"},
    
    {"folder" : "/home/aholaj/Data/EmulatorManuscriptData/LVL4Day/",
    "longID" : "LVL4Day",
    "ID_prefix" : "4D"},
    
    {"folder" : "/home/aholaj/Data/EmulatorManuscriptData/LVL4Night/",
    "longID" : "LVL4Night",
    "ID_prefix" : "4N"}
    ]
    
    for sanakirja in listOfDatasets:
        merge(sanakirja)
    
    
    
    
if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("Script completed in " + str(round((end - start),0)) + " seconds")
