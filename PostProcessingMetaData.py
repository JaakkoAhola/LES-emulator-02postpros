#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import pathlib
import sys
sys.path.append("../LES-03plotting")
from FileSystem import FileSystem

class PostProcessingMetaData:
    """
Created on Wed Dec 16 18:21:10 2020

@author: Jaakko Ahola, Finnish Meteorological Institute
@licence: MIT licence Copyright

PostProcessingMetaData Class
"""

    def __init__(self, name : str, trainingSimulationRootFolder : list, dataOutputRootFolder : list):
        
        self.name = name
        self.ID_prefix = name[3:5]
        
        self.trainingSimulationRootFolder = self.__joinFolders( trainingSimulationRootFolder )
        self.dataOutputRootFolder = self.__joinFolders( dataOutputRootFolder )
        
        self.dataOutputFolder = self.dataOutputRootFolder / self.name
        
        self.phase01CSVFile = self.dataOutputFolder / (self.name + "_phase01.csv")
        
        self.responseFromTrainingSimulationsCSVFile = self.dataOutputFolder / ( self.name + "_responseFromTrainingSimulations.csv")
        
        self.statsFile = self.dataOutputFolder / (self.name + "_stats.csv")
        
        self.dataFile = self.dataOutputFolder / (self.name + "_DATA")
        
        self.filteredFile = self.dataOutputFolder / (self.name + "_filtered.csv")
        
        self.completeFile = self.dataOutputFolder / (self.name + "_complete.csv")
        
        
        self.fortranDataFolder = self.dataOutputFolder / ( "DATA" + "_" + self.responseVariable)
        
        
        self.responseIndicatorVariable = "responseIndicator"
        
        self.testFolderExists(self.trainingSimulationRootFolder)
        
        FileSystem.makeFolder( self.dataOutputFolder )
        
        
    def __joinFolders(self, sequenceOfFolders : list):
        return pathlib.Path("/".join(sequenceOfFolders))
    
    def testFolderExists(self, folder):
        assert(folder.exists())

def main():
    pass
if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"\nScript completed in { end - start : .1f} seconds")
