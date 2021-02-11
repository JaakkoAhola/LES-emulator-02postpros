#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 17:31:36 2020

@author: Jaakko Ahola, Finnish Meteorological Institute
@licence: MIT licence Copyright

Second phase of post-processing simulation data for a emulator

Includes running actual emulator with leave-one-out method
"""
print(__doc__)
from copy import deepcopy
import itertools
import math
import multiprocessing
import numpy
import xarray
import pandas
import pathlib
import time
import os
import sys
from scipy import stats
import LES2emu
from subprocess import run
from sklearn.metrics import mean_squared_error
from itertools import repeat


try:
    import f90nml
    fortranModePossible = True
except ModuleNotFoundError:
    fortranModePossible = False


sys.path.append(os.environ["LESMAINSCRIPTS"])

from Data import Data
from InputSimulation import InputSimulation
from FileSystem import FileSystem

from PostProcessingMetaData import PostProcessingMetaData

sys.path.append(os.environ["PYTHONEMULATOR"])
from LeaveOneOut import LeaveOneOut

class EmulatorData(PostProcessingMetaData):

    def __init__(self, name : str,
                 trainingSimulationRootFolder : list,
                 dataOutputRootFolder : list,
                 configFile : str
                 ):
        # responseVariable : str,
        #          filteringVariablesWithConditions : dict,
        #          responseIndicator = 0
        super().__init__(name, trainingSimulationRootFolder, dataOutputRootFolder, configFile  = configFile)

        if self.useFortran:
            assert(fortranModePossible)
        
        self._readOptimizationConfigs()
        
        self.numCores = multiprocessing.cpu_count()

        self.anomalyLimitLow = {}
        self.anomalyLimitHigh = {}
        self.anomalyQuantile = 0.02

        self.tol_clw = 1e-5

        self.timeStart = 2.5
        self.timeEnd = 3.5

        self.toleranceForInEqualConditions = 0.1
        
        self.exeDataFolder = self.dataOutputFolder / ( "DATA" + "_" + self.responseVariable)

        if self.useFortran: FileSystem.makeFolder( self.exeDataFolder )
        
        self.__prepare__getDesignVariableNames()


    
    def _readOptimizationConfigs(self):
        try:
            self.optimization = self.configFile["optimization"]
            
            assert( "maxiter" in self.optimization.keys())
            assert( "n_restarts_optimizer" in self.optimization.keys())
            
        except KeyError:
            self.optimization = {"maxiter" : 15000,
                                 "n_restarts_optimizer" : 10}


    def setToleranceForInEqualConditions(self, tolerance):
        self.toleranceForInEqualConditions = tolerance

    def prepare(self):
        if self.completeFile.is_file():
            # File exists, lets override or read the file
            
            if self.override:
                print("File exists, but let's override")
                self._prepare_override()
            else:
                print("File exists, let's read it")
                self.simulationCompleteData = pandas.read_csv( self.completeFile, index_col = 0)
                if self.filterIndex in self.simulationCompleteData.columns:
                    self.__prepare_CompleteDataPreExisted()
                else:
                    self.__prepare_appendNewFilterIndexToCompleteData()
            
        else:
            print("File does not exist, let's create it")
            self._prepare_override()
    def __prepare_CompleteDataPreExisted(self):
        self.__prepare__filter()
        self.__prepare__getSimulationCollection()
            
    def __prepare_appendNewFilterIndexToCompleteData(self):
        self.__prepare__setResponseIndicatorCompleteData()
        
        self.__prepare__getSimulationCollection()
        
        self.__prepare__filterGetOutlierDataFromLESoutput()
        
        self.__prepare__filter()
        
        self._fillUpDrflxValues()

        if self.useFortran: self.__prepare__saveFilteredDataExeMode()

        self.__prepare__saveFilteredDataRMode()

        if self.useFortran: self.__prepare__saveOmittedData()

        if self.useFortran: self.__prepare__getExeFolderList()
        
        self.finalise()
            
    def _prepare_override(self):
                
        self.__prepare__getPhase01()

        self.__prepare__getResponseFromTrainingSimulations()

        self.__prepare__setSimulationCompleteDataFromDesignAndTraining()

        self.__prepare__setResponseIndicatorCompleteData()

        self.__prepare__getSimulationCollection()

        self.__prepare__filterGetOutlierDataFromLESoutput()

        self.__prepare__filter()
        
        self._fillUpDrflxValues()

        if self.useFortran: self.__prepare__saveFilteredDataExeMode()

        self.__prepare__saveFilteredDataRMode()

        if self.useFortran: self.__prepare__saveOmittedData()

        if self.useFortran: self.__prepare__getExeFolderList()
        
        self.finalise()

    def runEmulator(self):

        if self.useFortran:
            self.__runFortranEmulator()
        else:
            self.__runPythonEmulator()
        
        self.finalise()
        
    def __runFortranEmulator(self):

        self.__fortranEmulator__saveNamelists()

        self.__fortranEmulator__linkExecutables()

        self.__fortranEmulator__runTrain()

        self.__fortranEmulator__runPredictionParallel()

    def __runPythonEmulator(self):
        if self.runEmulator: self.__pythonEmulator__leaveOneOutPython()
        if self.runBootStrap: self.__pythonEmulator_bootstrap()

    def postProcess(self):
        
        if self.runPostProcessing:
            self._runPostProcess()
        else:
            print("Not postprocessing because runPostProcess set to False in configFile")
        
    def _runPostProcess(self):    
        if self.useFortran: self.collectSimulatedVSPredictedDataFortran()

        self._getAnomalyLimitsQuantile()
        self._getAnomalyLimitsConstant()

        self._getAnomalies()

        self._getLeaveOneOut()
        self._getLinearFit()

        self._getErrors()

        self._finaliseStats()
        self._finaliseAnomalies()

        self.finalise()



    def __prepare__getPhase01(self):
        assert(self.phase01CSVFile.is_file())
        self.phase01 = pandas.read_csv(self.phase01CSVFile, index_col=0)

    def __prepare__getDesignVariableNames(self):
        self.meteorologicalVariables = ["q_inv", "tpot_inv", "lwp", "tpot_pbl", "pblh"]

        self.microphysics = self.ID_prefix[0]

        self.timeOfDay = self.ID_prefix[1]

        if self.microphysics == "3":
            self.microphysicsVariables = ["cdnc"]
        elif self.microphysics == "4":
            self.microphysicsVariables = ["ks", "as", "cs", "rdry_AS_eff"]

        if self.timeOfDay == "D":
            self.timeOfDayVariable = ["cos_mu"]
        else:
            self.timeOfDayVariable = []

        self.designVariableNames = self.meteorologicalVariables + self.microphysicsVariables + self.timeOfDayVariable

    def __prepare__getResponseFromTrainingSimulations(self):

        if pathlib.Path( self.responseFromTrainingSimulationsCSVFile).is_file():
             self.responseFromTrainingSimulations = pandas.read_csv(self.responseFromTrainingSimulationsCSVFile, index_col=0)
        else:
             self.responseFromTrainingSimulations = LES2emu.GetEmu2Vars( str(self.trainingSimulationRootFolder), self.ID_prefix )

        self.responseFromTrainingSimulations.to_csv(self.responseFromTrainingSimulationsCSVFile)

        return self.responseFromTrainingSimulations

    def __prepare__setSimulationCompleteDataFromDesignAndTraining(self):

        self.simulationCompleteData = pandas.merge(self.phase01, self.responseFromTrainingSimulations,on = "ID", how="left")

    def __prepare__setResponseIndicatorCompleteData(self):
        self.simulationCompleteData[self.responseIndicatorVariable] = float(self.responseIndicator)

    def __prepare__filter(self):

        self.__filterAllConditions()

        self.simulationFilteredData = self.simulationCompleteData[ self.simulationCompleteData[self.filterIndex]]

        self.simulationFilteredData = self.simulationFilteredData[self.designVariableNames + [self.responseIndicatorVariable , self.responseVariable]]

        self.simulationFilteredCSV = self.simulationFilteredData[self.designVariableNames + [ self.responseVariable ]]

    def __filterAllConditions(self):
        self.__getConditions()
        self.simulationCompleteData[self.filterIndex] = Data.getAllConditions(self.conditions)

    def __getConditions(self):
        self.conditions = {}
        for key in self.filteringVariablesWithConditions:
            if "!=" in self.filteringVariablesWithConditions[key]:
                self.conditions[key] =  ~ (numpy.isclose( self.simulationCompleteData[ key ],
                                                                                     float(self.filteringVariablesWithConditions[key][2:]),
                                                                                     atol = self.toleranceForInEqualConditions ))
            else:
                self.conditions[key] = eval("self.simulationCompleteData." + key + self.filteringVariablesWithConditions[key])

    def __prepare__saveFilteredDataExeMode(self):

        self.simulationFilteredData.to_csv( self.dataFile, float_format="%017.11e", sep = " ", header = False, index = False, index_label = False )

    def __prepare__saveFilteredDataRMode(self):
        self.simulationFilteredCSV.to_csv(self.filteredFile)

    def __prepare__saveOmittedData(self):

        for ind in self.simulationFilteredData.index:
            folder = (self.exeDataFolder / str(ind) )
            folder.mkdir( parents=True, exist_ok = True )

            train = self.simulationFilteredData.drop(ind)
            train.to_csv( folder / ("DATA_train_" + ind ), float_format="%017.11e", sep = " ", header = False, index = False )

            predict = self.simulationFilteredData.loc[ind].values
            numpy.savetxt( folder / ("DATA_predict_" + ind), predict, fmt="%017.11e", newline = " ")

    def __fortranEmulator__saveNamelists(self):



        for ind in self.simulationFilteredData.index:
            folder = (self.exeDataFolder / str(ind) )
            folder.mkdir( parents=True, exist_ok = True )

            trainNML = {"inputoutput":
                    {"trainingDataInputFile":"DATA_train_" + ind,
                     "trainedEmulatorGPFile" :"out_" + ind + ".gp",
                     "separator": " ",
                     "debugFlag": False}}

            predictNML = {"inputoutput":
                      {"trainingDataInputFile":"DATA_train_" + ind,
	                   "trainedEmulatorGPFile":"out_" + ind + ".gp",
	                   "predictionDataInputFile":"DATA_predict_" + ind,
	                   "predictionOutputFile":"DATA_predict_output_" + ind,
	                   "separator":" ",
	                   "debugFlag":False}}

            with open(folder / "train.nml", mode="w") as trainNMLFile:
                f90nml.write(trainNML, trainNMLFile)
            with open(folder / "predict.nml", mode="w") as predictNMLFile:
                f90nml.write(predictNML, predictNMLFile)


    def __pythonEmulator__saveYAMLconfigFiles(self):

        for ind in self.simulationFilteredData.index:
            folder = (self.exeDataFolder / str(ind) )
            folder.mkdir( parents=True, exist_ok = True )

            configuration = {"trainingDataInputFile": str( folder / ("DATA_train_" + ind)),
                          "predictionDataInputFile" : str( folder / ("DATA_predict_" + ind)),
                          "predictionOutputFile" : str(folder / ("DATA_predict_output_" + ind)),}

            FileSystem.writeYAML(folder / "config.yaml", configuration)

    def __pythonEmulator__linkExecutables(self):
        for ind in self.simulationFilteredData.index:
            folder = (self.exeDataFolder / str(ind) )
            folder.mkdir( parents=True, exist_ok = True )

            pythonEmulatorFolder = pathlib.Path( os.environ["PYTHONEMULATOR"] )

            for pythonFile in pythonEmulatorFolder.glob("**/*.py"):
                run(["ln","-sf", pythonFile, folder / pythonFile.name ])
                
    def __pythonEmulator__leaveOneOutPython(self):
        
        print(f'Emulating {self.name}, maxiter: {self.optimization["maxiter"]}, n_restarts_optimizer: {self.optimization["n_restarts_optimizer"]}')
        t1 = time.time()
        
        with multiprocessing.Pool(processes = self.numCores) as pool:
            
            output = pool.starmap( LeaveOneOut.loopLeaveOneOut, zip(repeat(self.simulationFilteredData),
                                                                    self.simulationFilteredData.index.values,
                                                                    repeat(self.optimization),
                                                                    repeat(self.boundOrdo)) )
        leaveOneOutArray = numpy.asarray(output)
        
        rmse = mean_squared_error(self.simulationFilteredData[self.responseVariable].values, leaveOneOutArray, squared=False)
        
        t2 = time.time()
        timepercase = (t2 -t1) / (len(self.simulationFilteredData.index.values) )
        print(f"""
Emulating completed, {self.name}
Time emulated {t2-t1:.1f},
avg. {timepercase},
maxiter: {self.optimization['maxiter']},
n_restarts_optimizer: {self.optimization['n_restarts_optimizer']},
boundOrdo: self.boundOrdo,
rmse: {rmse:.2f}""")
        
        self.simulationFilteredData[self.emulatedVariable] = leaveOneOutArray
        
        self.simulationCompleteData = pandas.concat([ self.simulationCompleteData, self.simulationFilteredData[self.emulatedVariable] ], axis = 1)
            
    def __pythonEmulator_bootstrap(self):
        
        bootstrapRsquared = numpy.empty( self.bootstrappingParameters["iterations"] )
        bootstrapRMSE = numpy.empty( self.bootstrappingParameters["iterations"] )
        bootstrapAbsErrorSum = numpy.empty( self.bootstrappingParameters["iterations"] )
        bootstrapAbsErrorVar = numpy.empty( self.bootstrappingParameters["iterations"] )
        print("Bootstrapping")
        t1 = time.time()
        for bootStrapIndex in range(self.bootstrappingParameters["iterations"]):
            sampleDF = self.simulationFilteredData.sample(n = self.bootstrappingParameters["sampleSize"], random_state = bootStrapIndex).sort_index()
            
            
            with multiprocessing.Pool( processes = self.numCores ) as pool:
                output = pool.starmap( LeaveOneOut.loopLeaveOneOut, zip(repeat(sampleDF),
                                                                    sampleDF.index.values,
                                                                    repeat(self.optimization),
                                                                    repeat(self.boundOrdo)) )
            leaveOneOutArray = numpy.asarray( output )
                
            absError = numpy.abs( leaveOneOutArray - sampleDF[ self.responseVariable ].values)
            
            bootstrapAbsErrorSum[bootStrapIndex] = numpy.sum(absError)
            
            bootstrapAbsErrorVar[bootStrapIndex] = numpy.var(absError)
            
            bootstrapRMSE[bootStrapIndex] = mean_squared_error( sampleDF[ self.responseVariable ].values, leaveOneOutArray, squared = False  )
            
            slope, intercept, r_value, p_value, std_err = stats.linregress( sampleDF[ self.responseVariable ].values, leaveOneOutArray )
            
            bootstrapRsquared[bootStrapIndex] = numpy.power(r_value, 2)
        t2 = time.time()
        timepercase = (t2 -t1) / ( self.bootstrappingParameters["iterations"] * self.bootstrappingParameters["sampleSize"]  )
        print(f"""
Bootstrapping completed, {self.name}
Time bootstrapped {t2-t1:.1f},
avg. {timepercase},
iterations: {self.bootstrappingParameters["iterations"]},
sample size: {self.bootstrappingParameters["sampleSize"]}
""")
        
            
        self.bootstrapDataFrame = pandas.DataFrame(data = {"AbsErrorSum": bootstrapAbsErrorSum,
                                 "AbsErrorVar" :  bootstrapAbsErrorVar,
                                 "RMSE": bootstrapRMSE,
                                 "RSquared" : bootstrapRsquared})
        
        self.bootstrapDataFrame.to_csv( self.bootStrapFile )
            
        
            

    def __fortranEmulator__linkExecutables(self):
        for ind in self.simulationFilteredData.index:
            folder = (self.exeDataFolder / str(ind) )
            folder.mkdir( parents=True, exist_ok = True )
            run(["ln","-sf", os.environ["GPTRAINEMULATOR"], folder / "gp_train"])
            run(["ln","-sf", os.environ["GPPREDICTEMULATOR"], folder / "gp_predict"])

    def __prepare__getExeFolderList(self):
        indexGroup = self.simulationFilteredData.index.values

        self.exeFolderList = [ iterab[0] / iterab[1] for iterab in itertools.product([self.exeDataFolder], indexGroup) ]

    def getTrainingFileName(folder):
        ind = folder.name

        return pathlib.Path("out_" + ind + ".gp")

    def getPredictionInputFileName(folder):
        ind = folder.name

        return pathlib.Path("DATA_predict_" + ind)
    def getPredictionOutputFileName(folder):
        ind = folder.name

        return pathlib.Path("DATA_predict_output_" + ind)

    def __fortranEmulator__runTrain(self):

        try:
            pool = multiprocessing.Pool(processes = self.numCores)
            for ind in pool.imap_unordered( self._runTrain, deepcopy(self.exeFolderList)):
                 pass
        except Exception as e:
            print(e)
            pool.close()
        pool.close()

    def _runTrain(self, folder):

        os.chdir(folder)

        trainingFile = EmulatorData.getTrainingFileName(folder)

        trainingFileExists = trainingFile.is_file()

        if trainingFileExists:
            print(folder, "Training output file exists")
            trainingFileIsEmpty = os.stat(trainingFile).st_size == 0

            if trainingFileIsEmpty:
                trainingFile.unlink() #remove file

        if not trainingFileExists:
            print("runTrain", folder)
            run([ "./gp_train"])

    def __fortranEmulator__runPredictionParallel(self):

        try:
            pool = multiprocessing.Pool(processes = self.numCores)
            for ind in pool.imap_unordered( self._runPrediction, deepcopy(self.exeFolderList)):
                 pass
        except Exception as e:
            print(e)
            pool.close()
        pool.close()

    def runPredictionSerial(self):
        for folder in self.exeFolderList:
            self._runPrediction(folder)

    def _runPrediction(self,folder):
        os.chdir(folder)

        predictionFile = EmulatorData.getPredictionOutputFileName(folder)

        trainingFile = EmulatorData.getTrainingFileName(folder)

        if not trainingFile.is_file():
            raise Exception("Training output file does not exist. NOT predicting")

        fileExists = predictionFile.is_file()

        if fileExists:
            print(folder, "Prediction output exists")
            fileIsEmpty = numpy.isnan(self._readPredictedData("."))
            if fileIsEmpty:
                predictionFile.unlink() #remove file

        if not fileExists:
            print("runPrediction", folder)
            run([ "./gp_predict"])



    def collectSimulatedVSPredictedDataFortran(self):
        datadict = {self.simulatedVariable: [],
                    self.emulatedVariable: []}

        for folder  in self.exeFolderList:
            simulated = self._readSimulatedData(folder)
            emulated  = self._readPredictedData(folder)


            datadict[self.simulatedVariable].append(simulated)
            datadict[self.emulatedVariable].append(emulated)



        for key in datadict:

            self.simulationFilteredData[key] = datadict[key]

        self._checkIntegrety()

        self.simulationFilteredData.drop(columns = self.responseIndicatorVariable, inplace = True)
        self.simulationCompleteData.drop(columns = self.responseIndicatorVariable, inplace = True)

        if self.allIsWell:
            self.simulationFilteredData.drop(columns = self.simulatedVariable, inplace = True )

        self.simulationCompleteData = self.simulationCompleteData.merge( self.simulationFilteredData, on = ["ID", self.responseVariable] + self.designVariableNames, how = "left")

    def _readTrainingInputData(self, folder):
        try:
            data = pandas.read_csv( folder / EmulatorData.getTrainingFileName(folder), delim_whitespace = True, header = None)
        except pandas.errors.EmptyDataError:
            data = numpy.nan
        return data

    def _readSimulatedData(self, folder):
        try:
            data = pandas.read_csv( folder / EmulatorData.getPredictionInputFileName(folder), delim_whitespace = True, header = None).iloc[0,-1]
        except pandas.errors.EmptyDataError:
            data = numpy.nan
        return data

    def _readPredictionInputData(self, folder):
        try:
            data = pandas.read_csv( folder / EmulatorData.getPredictionInputFileName(folder), delim_whitespace = True, header = None).iloc[0,:-2]
        except pandas.errors.EmptyDataError:
            data = numpy.nan
        return data

    def _readPredictedData(self, folder):
        try:
            data = pandas.read_csv( folder / EmulatorData.getPredictionOutputFileName(folder), delim_whitespace = True, header = None).iloc[0,0]
        except pandas.errors.EmptyDataError:
            data = numpy.nan
        return data

    def _checkIntegrety(self):
        self.allIsWell = True
        for ind in self.simulationFilteredData.index:
            relativeError = numpy.abs( (self.simulationFilteredData.loc[ind][self.responseVariable] - self.simulationFilteredData.loc[ind][self.simulatedVariable])
                      / self.simulationFilteredData.loc[ind][self.responseVariable]) * 100.
            if relativeError > 0.1:
                print("case", ind, "relative error", relativeError)
                self.allIsWell = False

    def __prepare__getSimulationCollection(self):
        self.simulationCollection = InputSimulation.getSimulationCollection( self.simulationCompleteData )

    def __prepare__filterGetOutlierDataFromLESoutput(self):

        dataframe = self.simulationCompleteData
        collection = self.simulationCollection

        dataframe["lwpEndValue"] = dataframe.apply( lambda row: \
                                                   collection[ row["LABEL"] ].getTSDataset()["lwp_bar"].values[-1]*1000.,
                                                   axis = 1)

        dataframe["lwpRelativeChange"] = dataframe.apply( lambda row:\
                                                         row["lwpEndValue"] / row["lwp"],
                                                         axis = 1 )

        dataframe["cloudTopRelativeChange"] = dataframe.apply( lambda row: \
                                                              collection[ row["LABEL"] ].getTSDataset()["zc"].values[-1] / row["pblh_m"],
                                                              axis = 1)

        dataframe["cfracEndValue"] = dataframe.apply( lambda row: \
                                                     collection[ row["LABEL"] ].getTSDataset()["cfrac"].values[-1],
                                                     axis = 1)

    def _getAnomalyLimitsConstant(self):
        self.anomalyLimitLow["tpot_inv"] = 2.5
        self.anomalyLimitHigh["cloudTopRelativeChange"] = 1.1
        self.anomalyLimitHigh["lwpRelativeChange"] = 1.4

        self.anomalyLimitLow["tpot_pbl"] = 273
        self.anomalyLimitHigh["tpot_pbl"] =  300



        if self.name[4:] == "Day":
            self.anomalyLimitLow["cos_mu"] =  math.cos(5*math.pi/12)
            self.anomalyLimitHigh["cos_mu"] =  math.cos(math.pi/12)
        else:
            self.anomalyLimitLow["cos_mu"] = 0
            self.anomalyLimitHigh["cos_mu"] = 0

    def _getAnomalyLimitsQuantile(self):
        dataframe = self.simulationCompleteData

        self.anomalyLimitLow["tpot_inv"] = dataframe["tpot_inv"].quantile(self.anomalyQuantile)
        self.anomalyLimitHigh["cloudTopRelativeChange"] = dataframe["cloudTopRelativeChange"].quantile(1-self.anomalyQuantile)
        self.anomalyLimitHigh["lwpRelativeChange"] = dataframe["lwpRelativeChange"].quantile(1-self.anomalyQuantile)
        self.anomalyLimitLow["q_inv"] = dataframe["q_inv"].quantile(self.anomalyQuantile)

        self.anomalyLimitLow["pblh"] =  dataframe["pblh"].quantile(0.05)
        self.anomalyLimitHigh["pblh"] = dataframe["pblh"].quantile(0.95)

        self.anomalyLimitLow["lwp"] =  dataframe["lwp"].quantile(0.05)
        self.anomalyLimitHigh["lwp"] =  dataframe["lwp"].quantile(0.95)

        if self.name[3] == "3":
            self.anomalyLimitLow["cdnc"] = dataframe["cdnc"].quantile(0.05)
            self.anomalyLimitHigh["cdnc"] = dataframe["cdnc"].quantile(0.95)


    def _getAnomalies(self):
        dataframe = self.simulationCompleteData

        for key in self.anomalyLimitLow:
            try:
                dataframe[key + "_low_tail"] = dataframe[key] < self.anomalyLimitLow[key]
            except KeyError:
                pass
        for key in self.anomalyLimitHigh:
            try:
                dataframe[key + "_high_tail"] = dataframe[key] > self.anomalyLimitHigh[key]
            except KeyError:
                pass

    def _finaliseAnomalies(self):
        low = pandas.DataFrame(self.anomalyLimitLow, index = ["low"]).transpose()
        high = pandas.DataFrame(self.anomalyLimitHigh, index = ["high"]).transpose()

        self.anomalyLimitsCombined = pandas.concat([low,high], axis = 1, verify_integrity=True)

        self.anomalyLimitsCombined.to_csv(self.dataOutputRootFolder / "anomalyLimits.csv")


    def _fillUpDrflxValues(self):

        if self.ID_prefix == "3":
            return

        dataframe = self.simulationCompleteData

        if numpy.abs(dataframe["drflx"].max() - dataframe["drflx"].min() ) > 10*Data.getEpsilon():
            return

        newCloudRadiativeValues = numpy.zeros(numpy.shape(dataframe["drflx"]))

        for emulInd, emul in enumerate(list(self.simulationCollection)):
            self.simulationCollection[emul].getPSDataset()
            self.simulationCollection[emul].setTimeCoordToHours()

            psDataTimeSliced = self.simulationCollection[emul].sliceByTimePSDataset( self.timeStart, self.timeEnd )

            numberOfCloudyColumns = 0

            cloudRadiativeWarmingAllColumnValues = 0.

            for timeInd, timeValue in enumerate(psDataTimeSliced["time"]):
                rflxTimeSlice = psDataTimeSliced["rflx"].isel( time = timeInd )

                liquidWaterTimeSlice = psDataTimeSliced["l"].isel( time = timeInd )

                psDataCloudyPointIndexes, = numpy.where( liquidWaterTimeSlice > self.tol_clw )
                if len(psDataCloudyPointIndexes > 0):
                    numberOfCloudyColumns += 1

                    firstCloudyGridCell = psDataCloudyPointIndexes[0]
                    lastCloudyGridCell = psDataCloudyPointIndexes[-1]


                    cloudRadiativeWarmingAllColumnValues += rflxTimeSlice[firstCloudyGridCell] - rflxTimeSlice[lastCloudyGridCell]

            ## end time for loop
            if numberOfCloudyColumns > 0:
                drflx = cloudRadiativeWarmingAllColumnValues.values / numberOfCloudyColumns
            else:
                drflx = 0.

            newCloudRadiativeValues[emulInd] = drflx

        ### end emul for loop
        dataframe["drflx"] = newCloudRadiativeValues

    def _getWeightedUpdraft(self):

        dataframe = self.simulationCompleteData

        dataframe["wposWeighted"] = numpy.zeros(numpy.shape(dataframe["wpos"]))

        print("Start calculation of weighted updrafts " + self.name)

        t1 = time.time()

        for emul in self.simulationCollection:

            filename = self.simulationCollection[emul].getNCDatasetFileName()
            ncData = xarray.open_dataset(filename)
            timeStartInd = Data.getClosestIndex( ncData.time.values, self.timeStart*3600. )
            timeEndInd   = Data.getClosestIndex( ncData.time.values, self.timeEnd*3600. )+1

            ncDataSliced = ncData.isel(time=slice(timeStartInd, timeEndInd))

            cloudMaskHalf = self.__getCloudMask( ncDataSliced["l"].values, self.__maskCloudColumnUpToBottomHalfOfCloud )

            wPosValuesHalfMaskedNANIncluded = ncDataSliced["w"].where(ncDataSliced["w"] > 0. ).values[ cloudMaskHalf ]

            wPosValuesHalfMasked = wPosValuesHalfMaskedNANIncluded[ numpy.logical_not(numpy.isnan(wPosValuesHalfMaskedNANIncluded)) ]

            weighted = numpy.sum( numpy.power( wPosValuesHalfMasked, 2 ) ) / numpy.sum( wPosValuesHalfMasked )

            dataframe.loc[ dataframe.index == emul, "wposWeighted"] = weighted
        t2 = time.time()

        print(f"Time to calculate updrafts {t2-t1:.1f}")



    def __getCloudMask(self, cloudWaterMatrix, maskingFunction ):
        listOfZColumnArrays, originalShape = self.__getListOfZArrays( cloudWaterMatrix )

        vectorizedFunction = numpy.vectorize( maskingFunction, signature = "(n)->(n)" )

        maskedZColumnArrays = vectorizedFunction( listOfZColumnArrays )

        cloudMask = numpy.reshape(maskedZColumnArrays, originalShape)

        return cloudMask

    def __getListOfZArrays(self, arr):
        originalShape = arr.shape
        listOfZArrays = numpy.reshape(arr, (numpy.prod(arr.shape[:3]), arr.shape[3]))

        return listOfZArrays, originalShape


    def __maskCloudColumnUpToBottomHalfOfCloud(self, zarr):
        maskedZarr = numpy.zeros( numpy.shape(zarr), dtype = bool )
        cloudyPoints = numpy.where( zarr > self.tol_clw )[0]
        if len(cloudyPoints)> 0:
            lowestCloudyZPointInColumn = cloudyPoints[0]
            highestCloudyZPointInColumn = cloudyPoints[-1]
            maskedZarr[ lowestCloudyZPointInColumn : ((highestCloudyZPointInColumn - lowestCloudyZPointInColumn)//2 + lowestCloudyZPointInColumn+1) ] = 1

        return maskedZarr

    def __maskCloudColumnLowestGridPoint(self, zarr):
        maskedZarr = numpy.zeros( numpy.shape(zarr), dtype = bool )
        cloudyPoints = numpy.where( zarr > self.tol_clw )[0]
        if len(cloudyPoints)> 0:
            lowestCloudyZPointInColumn = cloudyPoints[0]
            maskedZarr[ lowestCloudyZPointInColumn ] = 1

        return maskedZarr


    def _getLeaveOneOut(self):
        dataframe = self.simulationCompleteData

        dataframe["leaveOneOutIndex"]  = dataframe[self.filterIndex]

        dataframe = dataframe.loc[dataframe["leaveOneOutIndex"]]

        simulated = dataframe[ self.responseVariable ].values
        emulated  = dataframe[ self.emulatedVariable].values


        slope, intercept, r_value, p_value, std_err = stats.linregress(simulated, emulated)

        rSquared = numpy.power(r_value, 2)

        self.leaveOneOutStats = [slope, intercept, r_value, p_value, std_err, rSquared]


    def _getLinearFit(self):

        dataframe = self.simulationCompleteData

        dataframe["linearFitIndex"] = dataframe[self.filterIndex]
        
        dataframe = dataframe.loc[dataframe["linearFitIndex"]]

        radiativeWarming  = dataframe["drflx"].values
        updraft =  dataframe[ self.responseVariable ].values

        slope, intercept, r_value, p_value, std_err = stats.linregress(radiativeWarming, updraft)

        rSquared = numpy.power(r_value, 2)

        self.linearFitStats = [slope, intercept, r_value, p_value, std_err, rSquared]

        coef = [slope, intercept]
        poly1d_fn = numpy.poly1d(coef)

        self.simulationCompleteData[ self.linearFitVariable ] = self.simulationCompleteData.apply(lambda row: poly1d_fn(row["drflx"]), axis = 1)

    def _getErrors(self):
        dataframe = self.simulationCompleteData

        dataframe["absErrorEmul"] = dataframe.apply(lambda row: row[self.responseVariable] - row[self.emulatedVariable], axis = 1)

        dataframe["absErrorLinearFit"] = dataframe.apply(lambda row: row[self.responseVariable] - row[self.linearFitVariable], axis = 1)

    def _finaliseStats(self):
        self.statsDataFrame = pandas.DataFrame(numpy.array([  self.leaveOneOutStats, self.linearFitStats ]),
                                               columns = ["slope", "intercept", "r_value", "p_value", "std_err", "rSquared"],
                                               index=["leaveOneOutStats", "linearFitStats"])

        self.statsDataFrame.to_csv( self.statsFile)

    def finalise(self):
        self.simulationCompleteData.to_csv(self.completeFile)
