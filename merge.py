#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 14:58:24 2020

@author: Jaakko Ahola, Finnish Meteorological Institute
@licence: MIT licence Copyright
"""
import time
import pandas
import pathlib


class Merge:
    def __init__(self, name : str, dataOutputRootFolder : str):
        
        self.name = name
        
        self.ID_prefix = name[3:5] 
        
        self.rootFolder = pathlib.Path(dataOutputRootFolder)
        
        self.subFolder = self.rootFolder / name
        
        
        self.responseFile = self.subFolder / (name + "_responseFromTrainingSimulations.csv")
        
        self.simulatedVSPredictedFile = self.subFolder / (name + "_simulatedVSPredictedData.csv")
        
        self._initMetaData()
        
        self._initResponse()
        
        self._initSimulatedVSPredictedFile()
        
        
        
    def _initMetaData(self):
        self.metaDataFrame = pandas.read_csv(self.rootFolder / (self.name + ".csv"))
        
        self.metaDataFrame = self.metaDataFrame.drop(columns= ["ID.1", "ID.1.1"])
        
    def _initResponse(self):
        self.responseDataFrame = pandas.read_csv(self.responseFile)
        
        self.responseDataFrame["ID"] = self.responseDataFrame.apply(lambda row: self.ID_prefix +"_" + str(int(row["i"])).zfill(3), axis = 1)
        
        self.responseDataFrame = self.responseDataFrame.drop(columns="i")
        
    def _initSimulatedVSPredictedFile(self):
        self.simulatedVSPredictedDataFrame = pandas.read_csv(self.simulatedVSPredictedFile)
        
        self.simulatedVSPredictedDataFrame["ID"] = self.simulatedVSPredictedDataFrame.apply( \
                                                            lambda row: self.ID_prefix +"_" + str(int(row["designCase"] + 1)).zfill(3), axis = 1)
        
        self.simulatedVSPredictedDataFrame = self.simulatedVSPredictedDataFrame.drop(columns="designCase")
        
    def merge(self):
        
        self.mergedDataFrame = self.metaDataFrame.merge( self.responseDataFrame, on="ID", how = "left")
        
        self.mergedDataFrame = self.mergedDataFrame.merge(self.simulatedVSPredictedDataFrame,
                                                          on = ["ID", "q_inv", "tpot_inv", "lwp", "tpot_pbl", "pblh", "cdnc", "wpos"],
                                                          how="left")

        self.mergedDataFrame.set_index("ID", inplace = True )
        
        
        
        
    def saveMerged(self):
        
        merged.to_csv( self.rootFolder /  (self.name +  "_merged.csv"), index_label = "ID")
        
    
    def main():
        
        rootFolderOfDataOutputs = os.environ["EMULATORPOSTPROSDATAROOTFOLDER"]
   
    
        allData = {"LVL3Night" : Merge("LVL3Night",
                                             rootFolderOfDataOutputs,
                                             ),
                "LVL3Day"   :  Merge( "LVL3Day",
                                            rootFolderOfDataOutputs,
                                            responseVariable
                                            ),
                "LVL4Night" :  Merge("LVL4Night",
                                            rootFolderOfDataOutputs,
                                            responseVariable
                                            ),
                "LVL4Day"   : Merge("LVL4Day",
                                           rootFolderOfDataOutputs,
                                           responseVariable
                                           )
                }
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
        
    
    
    
    
if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("Script completed in " + str(round((end - start),0)) + " seconds")
