#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 17:31:36 2020

@author: Jaakko Ahola, Finnish Meteorological Institute
@licence: MIT licence Copyright
"""
import numpy
import pandas
import pathlib
import time

import LES2emu

class EmulatorData:
    def __init__(self, name, trainingSimulationRootFolder, dataOutputRootFolder, trainingOutputVariable, filterValue = -999, responseIndicator = 0):
        self.name = name
        self.trainingSimulationRootFolder = pathlib.Path( trainingSimulationRootFolder )
        self.dataOutputRootFolder = pathlib.Path(dataOutputRootFolder)
        self.trainingOutputVariable = trainingOutputVariable
        
        self.designCSVFile = self.dataOutputRootFolder / (self.name + "_design.csv")
        self.trainingOutputCSVFile = self.dataOutputRootFolder / ( self.name + "_" + self.trainingOutputVariable + ".csv")
        self.filteredCSVFile = self.dataOutputRootFolder / (self.name + "_filtered.csv")
        
        self.dataFile = self.dataOutputRootFolder / (self.name + "_DATA")
        
        self.filterValue = filterValue
        self.responseIndicator = responseIndicator
        
        self.fortranDataFolder = self.dataOutputRootFolder / "DATA"
        
    
    def getName(self):
        return self.name
    
    def getTrainingSimulationRootFolder(self):
        return self.trainingSimulationRootFolder
    
    def getDataOutputFolder(self):
        return self.dataOutputRootFolder
    
    def getTrainingOutputVariable(self):
        return self.trainingOutputVariable
    
    def getDesignCSVFile(self):
        return self.designCSVFile
    
    def getTrainingOutputCSVFile(self):
        return self.trainingOutputCSVFile
    
    def getSimulationCompleteData(self):
        return self.simulationCompleteData
    
    def getSimulationFilteredData(self):
        return self.simulationFilteredData
    
    def saveUpdraftFromTrainingSimulations(self):
        #,emulatorFolder, csvFolder, csvFilename
        
        EmuVars = LES2emu.GetEmu2Vars( str(self.trainingSimulationRootFolder) )
        
        with open( self.trainingOutputCSVFile,'w') as file:
            file.writelines(["%s\n" % (item[7])  for item in EmuVars])

    def saveFilteredData(self):
        self.simulationFilteredData.to_csv( self.filteredCSVFile )

    
    def setResponseIndicatorCompleteData(self):
        self.simulationCompleteData = self._setResponseIndicator( self.simulationCompleteData )
        
    def setResponseIndicatorFilteredData(self):
        self.simulationFilteredData = self._setResponseIndicator( self.simulationFilteredData )
    
    def _setResponseIndicator(self, data):
        data.insert( loc = len(data.columns) - 1, column = "responseIndicator", value = ( numpy.ones( data.shape[0] ) * self.responseIndicator).astype( numpy.float ))
        
        return data
    
    def saveFilteredDataFortranMode(self):
        #
        self.simulationFilteredData.to_csv( self.dataFile, float_format="%017.11e", sep = " ", header = False, index = False, index_label = False )
    
    def saveOmittedData(self):
        
        self.simulationFilteredData = self.simulationFilteredData.set_index( numpy.arange( self.simulationFilteredData.shape[0] ) )
        
        for ind in range(self.simulationFilteredData.shape[0]):
            folder = (self.fortranDataFolder / str(ind) )
            folder.mkdir( parents=True, exist_ok = True )
            
            train = self.simulationFilteredData.drop(ind, axis = 0)
            train.to_csv( folder / "DATA_train", sep = " ", header = False, index = False )
            
            predict = self.simulationFilteredData.iloc[ind]
            
            predict.to_csv( folder / "DATA_predict", sep = " ", header = False, index = False, line_terminator = " ")
    
    def setSimulationCompleteDataFromDesignAndTraining(self):
        #designCSVPath, trainingCSVPath, trainingVariableName = "wpos"
        
        design = pandas.read_csv( self.designCSVFile )
        
        training = pandas.read_csv( self.trainingOutputCSVFile, header=None )
        
        design[ self.trainingOutputVariable ] = training
        
        del design["Unnamed: 0"]
        
        self.simulationCompleteData = design
        
    def filterNan(self):
        
        self.simulationFilteredData = self.simulationCompleteData[ self.simulationCompleteData[ self.trainingOutputVariable ] != self.filterValue ]
        
    
    
        
    
def main():
    
    rootFolderOfEmulatorSets = "/home/aholaj/mounttauskansiot/eclairmount/"
    rootFolderOfDataOutputs = "/home/aholaj/Nextcloud/000_WORK/000_ARTIKKELIT/001_Manuscript_LES_emulator/data/"
    trainingOutputVariable = "wpos"
    getTrainingOutputFLAG = False
   
    ###########
    
    rootFolderOfEmulatorSets = pathlib.Path(rootFolderOfEmulatorSets)
    rootFolderOfDataOutputs = pathlib.Path(rootFolderOfDataOutputs)
    
    emulatorSets = {"LVL3Night" : EmulatorData("LVL3Night",
                                             rootFolderOfEmulatorSets /  "case_emulator_DESIGN_v3.0.0_LES_ECLAIR_branch_ECLAIRv2.0.cray.fast_LVL3_night",
                                             rootFolderOfDataOutputs / "LVL3Night",
                                             trainingOutputVariable
                                             ),
                "LVL3Day"   :  EmulatorData( "LVL3Day",
                                            rootFolderOfEmulatorSets / "case_emulator_DESIGN_v3.1.0_LES_ECLAIR_branch_ECLAIRv2.0.cray.fast_LVL3_day",
                                            rootFolderOfDataOutputs / "LVL3Day",
                                            trainingOutputVariable
                                            ),
                "LVL4Night" :  EmulatorData("LVL4Night",
                                            rootFolderOfEmulatorSets / "case_emulator_DESIGN_v3.2_LES_ECLAIR_branch_ECLAIRv2.0.cray.fast_LVL4_night",
                                            rootFolderOfDataOutputs / "LVL4Night",
                                            trainingOutputVariable
                                            ),
                "LVL4Day"   : EmulatorData("LVL4Day",
                                           rootFolderOfEmulatorSets / "case_emulator_DESIGN_v3.3_LES_ECLAIR_branch_ECLAIRv2.0.cray.fast_LVL4_day",
                                           rootFolderOfDataOutputs / "LVL4Day",
                                           trainingOutputVariable
                                           )
                }
    
    for ind, key in enumerate(list(emulatorSets)):
        
        if getTrainingOutputFLAG: emulatorSets[key].saveUpdraftFromTrainingSimulations()
        
        emulatorSets[key].setSimulationCompleteDataFromDesignAndTraining()
        
        emulatorSets[key].filterNan()
        emulatorSets[key].saveFilteredData()
        
        emulatorSets[key].setResponseIndicatorFilteredData()
        
        emulatorSets[key].saveFilteredDataFortranMode()
        
        emulatorSets[key].saveOmittedData()
    
    #print(emulatorSets["LVL3Day"].getSimulationFilteredData())
    
if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("Script completed in " + str(round((end - start),0)) + " seconds")
