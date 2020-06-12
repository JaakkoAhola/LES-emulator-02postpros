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
import os
import sys
import LES2emu
import f90nml
from subprocess import run
import multiprocessing
import functools
from shutil import copyfile

class EmulatorData:
    def __init__(self, name, trainingSimulationRootFolder, dataOutputRootFolder, trainingOutputVariable, filterValue = -999, responseIndicator = 0):
        self.name = name
        self.trainingSimulationRootFolder = pathlib.Path( trainingSimulationRootFolder )
        self.dataOutputFolder = pathlib.Path(dataOutputRootFolder) / self.name
        self.trainingOutputVariable = trainingOutputVariable
        
        self.possibleEmulatedVariables = ["id", "cond", "sedi", "coag", "auto", "diag", "prcp", "wpos", "w2pos", "cdnc_p", "cdnc_wp", "n"]
        
        self.possibleEmulatedVariablesDict = dict(zip(self.possibleEmulatedVariables, range(len(self.possibleEmulatedVariables))))
        
        self.indexOfTrainingOutputVariable = self.possibleEmulatedVariablesDict[ self.trainingOutputVariable ]
        
        self.designCSVFile = self.dataOutputFolder / (self.name + "_design.csv")
        self.trainingOutputCSVFile = self.dataOutputFolder / ( self.name + "_" + self.trainingOutputVariable + ".csv")
        self.filteredCSVFile = self.dataOutputFolder / (self.name + "_filtered.csv")
        
        self.dataFile = self.dataOutputFolder / (self.name + "_DATA")
        
        self.filterValue = filterValue
        self.responseIndicator = responseIndicator
        
        self.fortranDataFolder = self.dataOutputFolder / ( "DATA" + "_" + self.trainingOutputVariable)
        
        self.numCores = multiprocessing.cpu_count()
    
    def _getFilteredSize(self):
        
        try:
            returnValue = self.simulationFilteredData.shape[0]
        except AttributeError:
            try:
                pathGlob = self.fortranDataFolder.glob("**/*")
                returnValue = len([x for x in pathGlob if x.is_dir()])
            except:
                sys.exit("_getFilteredSize not working")
        
        return returnValue
        
    
    def getName(self):
        return self.name
    
    def getTrainingSimulationRootFolder(self):
        return self.trainingSimulationRootFolder
    
    def getDataOutputFolder(self):
        return self.dataOutputFolder
    
    def getTrainingOutputVariable(self):
        return self.trainingOutputVariable
    
    def getDesignCSVFile(self):
        return self.designCSVFile
    
    def getTrainingOutputCSVFile(self):
        return self.trainingOutputCSVFile
    
    def getSimulationCompleteData(self):
        return self.simulationCompleteData
    
    def getSimulationFilteredData(self):
        try:
            return self.simulationFilteredData
        except AttributeError:
            self.simulationFilteredData = pandas.read_csv( self.filteredCSVFile )
            return self.simulationFilteredData
    
    def copyDesignToPostProsFolder(self):
        copyfile(self.trainingSimulationRootFolder / "design.csv" ,  self.designCSVFile)
    
    def saveResponseFromTrainingSimulations(self):
        #,emulatorFolder, csvFolder, csvFilename
        
        EmuVars = LES2emu.GetEmu2Vars( str(self.trainingSimulationRootFolder) )
        
        with open( self.trainingOutputCSVFile,'w') as file:
            file.writelines(["%s\n" % (item[self.indexOfTrainingOutputVariable])  for item in EmuVars])

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
            train.to_csv( folder / "DATA_train", float_format="%017.11e", sep = " ", header = False, index = False )
            
            predict = self.simulationFilteredData.iloc[ind].values
            numpy.savetxt( folder / "DATA_predict", predict, fmt="%017.11e", newline = " ")
            
    def saveNamelists(self):
        trainNML = {"inputoutput":
                    {"trainingDataInputFile":"DATA_train",
                     "trainedEmulatorGPFile" :"out.gp",
                     "separator": " ",
                     "debugFlag": False}}
            
        predictNML = {"inputoutput":
                      {"trainingDataInputFile":"DATA_train",
	                   "trainedEmulatorGPFile":"out.gp",
	                   "predictionDataInputFile":"DATA_predict",
	                   "predictionOutputFile":"DATA_predict_output",
	                   "separator":" ",
	                   "debugFlag":False}}
        
        
        for ind in range(self._getFilteredSize()):
            folder = (self.fortranDataFolder / str(ind) )
            folder.mkdir( parents=True, exist_ok = True )
            with open(folder / "train.nml", mode="w") as trainNMLFile:
                f90nml.write(trainNML, trainNMLFile)
            with open(folder / "predict.nml", mode="w") as predictNMLFile:
                f90nml.write(predictNML, predictNMLFile)
                
                
    def linkExecutables(self):
        for ind in range(self._getFilteredSize()):
            folder = (self.fortranDataFolder / str(ind) )
            folder.mkdir( parents=True, exist_ok = True )
            run(["ln","-sf", os.environ["GPTRAINEMULATOR"], folder / "gp_train"])
            run(["ln","-sf", os.environ["GPPREDICTEMULATOR"], folder / "gp_predict"])
    
        
        
    def _runTrain(self, ind):
        
        folder = (self.fortranDataFolder / str(ind) )
        print(" ")
        print(self.name,"checking training output, case", ind)
        os.chdir(folder)
        
        if not pathlib.Path("out.gp").is_file():
            print("runTrain", folder)
            run([ "./gp_train"])
    
    def runTrain(self):
        
         
        indexGroup = range(self._getFilteredSize())
        # partialSelf = functools.partial(self._runTrain)
        try:
            pool = multiprocessing.Pool(processes = self.numCores)
            for ind in pool.imap_unordered( self._runTrain, indexGroup):
                 pass     
        except Exception as e:
            pool.close()
        pool.close()
    
    def _runPrediction(self,ind):
        folder = (self.fortranDataFolder / str(ind) )
        print(" ")
        print(self.name, "checking prediction output, case", ind)
        os.chdir(folder)
        outputfile = pathlib.Path("DATA_predict_output")
        
        fileExists = outputfile.is_file()
        
        if fileExists:
            fileIsEmpty = numpy.isnan(self._readPredictedData("."))
            if fileIsEmpty:
                outputfile.unlink() #remove file
            
        if not fileExists:
            print("runPrediction", folder)
            run([ "./gp_predict"])
    
    def runPredictionParallel(self):
        
        indexGroup = range(self._getFilteredSize())
        
        try:
            pool = multiprocessing.Pool(processes = self.numCores)
            for ind in pool.imap_unordered( self._runPrediction, indexGroup):
                 pass
        except Exception as e:
            pool.close()
        pool.close()
        
    def runPredictionSerial(self):
        for ind in range(self._getFilteredSize()):
            self._runPrediction(ind)
        
            
    def collectSimulatedVSPredictedData(self):
        datadict = {self.trainingOutputVariable + "_Simulated": [],
                    self.trainingOutputVariable + "_Emulated": []}
        
        self.simulatedVSPredictedData = self.getSimulationFilteredData()
        
        for ind in range(self._getFilteredSize()):
            folder = (self.fortranDataFolder / str(ind) )
            
            simulated = pandas.read_csv( folder / "DATA_predict", delim_whitespace = True, header = None).iloc[0,-1]
            emulated  = self._readPredictedData(folder)
            

            datadict[self.trainingOutputVariable + "_Simulated"].append(simulated)
            datadict[self.trainingOutputVariable + "_Emulated"].append(emulated)
        
        
        for key in datadict:
            
            self.simulatedVSPredictedData[key] = datadict[key]
        
        self.simulatedVSPredictedData.rename(columns={'Unnamed: 0': 'designCase'}, inplace=True)
        
        self.simulatedVSPredictedData.to_csv( self.dataOutputFolder / (self.name + "_simulatedVSPredictedData.csv"),
                                             sep = ",", index = False )
        
    
    def setSimulationCompleteDataFromDesignAndTraining(self):
        #designCSVPath, trainingCSVPath, trainingVariableName = "wpos"
        
        design = pandas.read_csv( self.designCSVFile )
        
        training = pandas.read_csv( self.trainingOutputCSVFile, header=None )
        
        design[ self.trainingOutputVariable ] = training
        
        del design["Unnamed: 0"]
        
        self.simulationCompleteData = design
        
    def filterNan(self):
        
        self.simulationFilteredData = self.simulationCompleteData[ self.simulationCompleteData[ self.trainingOutputVariable ] != self.filterValue ]
        
    def _readPredictedData(self, folder):
        try:
            data = pandas.read_csv( pathlib.Path(folder) / "DATA_predict_output", delim_whitespace = True, header = None).iloc[0,0]
        except pandas.errors.EmptyDataError:
            data = numpy.nan
        return data
    
    
        
    
def main(trainingOutputVariable):
    
    rootFolderOfEmulatorSets = os.environ["EMULATORDATAROOTFOLDER"]
    rootFolderOfDataOutputs = os.environ["EMULATORPOSTPROSDATAROOTFOLDER"]
    getTrainingOutputFLAG = False
   
    ###########
    
    rootFolderOfEmulatorSets = pathlib.Path(rootFolderOfEmulatorSets)
    rootFolderOfDataOutputs = pathlib.Path(rootFolderOfDataOutputs)
    
    emulatorSets = {"LVL3Night" : EmulatorData("LVL3Night",
                                             rootFolderOfEmulatorSets /  "case_emulator_DESIGN_v3.0.0_LES_ECLAIR_branch_ECLAIRv2.0.cray.fast_LVL3_night",
                                             rootFolderOfDataOutputs,
                                             trainingOutputVariable
                                             ),
                "LVL3Day"   :  EmulatorData( "LVL3Day",
                                            rootFolderOfEmulatorSets / "case_emulator_DESIGN_v3.1.0_LES_ECLAIR_branch_ECLAIRv2.0.cray.fast_LVL3_day",
                                            rootFolderOfDataOutputs,
                                            trainingOutputVariable
                                            ),
                "LVL4Night" :  EmulatorData("LVL4Night",
                                            rootFolderOfEmulatorSets / "case_emulator_DESIGN_v3.2_LES_ECLAIR_branch_ECLAIRv2.0.cray.fast_LVL4_night",
                                            rootFolderOfDataOutputs,
                                            trainingOutputVariable
                                            ),
                "LVL4Day"   : EmulatorData("LVL4Day",
                                           rootFolderOfEmulatorSets / "case_emulator_DESIGN_v3.3_LES_ECLAIR_branch_ECLAIRv2.0.cray.fast_LVL4_day",
                                           rootFolderOfDataOutputs,
                                           trainingOutputVariable
                                           )
                }
    
    for key in emulatorSets:

        if getTrainingOutputFLAG: emulatorSets[key].saveResponseFromTrainingSimulations()
        
        emulatorSets[key].copyDesignToPostProsFolder()
        
        emulatorSets[key].setSimulationCompleteDataFromDesignAndTraining()
        
        emulatorSets[key].filterNan()
        emulatorSets[key].saveFilteredData()
        
        emulatorSets[key].setResponseIndicatorFilteredData()
        
        emulatorSets[key].saveFilteredDataFortranMode()
        
        emulatorSets[key].saveOmittedData()
        
        emulatorSets[key].saveNamelists()
        
        emulatorSets[key].linkExecutables()
        
        emulatorSets[key].runTrain()
        
        emulatorSets[key].runPredictionSerial()
        
        emulatorSets[key].collectSimulatedVSPredictedData()
    
    
if __name__ == "__main__":
    start = time.time()
    try:
        trainingOutputVariable = sys.argv[1]
    except IndexError:
        trainingOutputVariable = "wpos"
    
    print("trainingOutputVariable: ",trainingOutputVariable)
    main(trainingOutputVariable)
    end = time.time()
    print("Script completed in " + str(round((end - start),0)) + " seconds")
