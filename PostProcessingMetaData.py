#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import pathlib
import sys
import os

from EmulatorMetaData import EmulatorMetaData

sys.path.append(os.environ["LESMAINSCRIPTS"])
from FileSystem import FileSystem


class PostProcessingMetaData(EmulatorMetaData):
    """
Created on Wed Dec 16 18:21:10 2020

@author: Jaakko Ahola, Finnish Meteorological Institute
@licence: MIT licence Copyright

PostProcessingMetaData Class
"""
    def __init__(self, name : str, trainingSimulationSubFolder : str, locationsFile : str):

        self.name = name
        self.ID_prefix = name[3:5]
        
        super().__init__(locationsFile)
            

        self.trainingSimulationRootFolder = self.trainingSimulationRootFolder / trainingSimulationSubFolder

        self.dataOutputFolder = self.postProsDataRootFolder / self.name

        self.phase01CSVFile = self.dataOutputFolder / (self.name + "_phase01.csv")

        self.responseFromTrainingSimulationsCSVFile = self.dataOutputFolder / ( self.name + "_responseFromTrainingSimulations.csv")

        self.statsFile = self.dataOutputFolder / (self.name + "_stats.csv")
        
        self.bootStrapFile = self.dataOutputFolder / (self.name + "_bootstrap.csv")
        
        self.bootstrapReplacementFile = self.dataOutputFolder / (self.name + "_bootstrapReplacement.csv")

        self.dataFile = self.dataOutputFolder / (self.name + "_DATA")

        self.filteredFile = self.dataOutputFolder / (self.name + "_filtered.csv")

        self.completeFile = self.dataOutputFolder / (self.name + "_complete.csv")
        
        self.featureImportanceFile = self.dataOutputFolder / (self.name + "_featureImportance.csv")

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
