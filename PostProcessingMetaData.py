#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 18:21:10 2020

@author: Jaakko Ahola, Finnish Meteorological Institute
@licence: MIT licence Copyright

PostProcessingMetaData Class
"""
import time
import pathlib
class PostProcessingMetaData:
    def __init__(self, name : str, trainingSimulationRootFolder : list, dataOutputRootFolder : list):
        
        self.name = name
        self.ID_prefix = name[3:5]
        
        self.trainingSimulationRootFolder = self.__joinFolders( trainingSimulationRootFolder )
        self.dataOutputRootFolder = self.__joinFolders( dataOutputRootFolder )
        
        self.dataOutputFolder = self.dataOutputRootFolder / self.name
        
        self.phase01CSVFile = self.dataOutputFolder / (self.name + "_phase01.csv")
        
        self.testFolderExists(self.trainingSimulationRootFolder)
        
    def __joinFolders(self, sequenceOfFolders : list):
        return pathlib.Path("/".join(sequenceOfFolders))
    
    def testFolderExists(self, folder):
        assert(folder.exists())
    def __doc__():
        print(__doc__)

def main():
    pass
if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"\nScript completed in { end - start : .1f} seconds")
