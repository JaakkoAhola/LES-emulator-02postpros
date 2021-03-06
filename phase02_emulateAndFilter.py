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
import f90nml
from subprocess import run


sys.path.append(os.environ["LESMAINSCRIPTS"])
from Data import Data
from InputSimulation import InputSimulation
from FileSystem import FileSystem

from PostProcessingMetaData import PostProcessingMetaData

class EmulatorData(PostProcessingMetaData):

    def __init__(self, name : str,
                 trainingSimulationRootFolder : list,
                 dataOutputRootFolder : list,
                 inputYAMLFile : str
                 ):
        # responseVariable : str,
        #          filteringVariablesWithConditions : dict,
        #          responseIndicator = 0
        super().__init__(name, trainingSimulationRootFolder, dataOutputRootFolder)

        self.inputYAMLFile = inputYAMLFile

        self._handleYAML()

        self.responseVariable = self.yamlDictionary["responseVariable"]

        self.useFortran = self.yamlDictionary["useFortran"]

        self.simulatedVariable = self.responseVariable + "_Simulated"

        self.emulatedVariable = self.responseVariable + "_Emulated"

        self.linearFitVariable = self.responseVariable + "_LinearFit"

        self.filteringVariablesWithConditions = self.yamlDictionary["filteringVariablesWithConditions"]

        self.testIfResponseVariableIsInFilter()

        self.responseIndicator = self.yamlDictionary[self.responseIndicatorVariable]

        self.numCores = multiprocessing.cpu_count()

        self.anomalyLimitLow = {}
        self.anomalyLimitHigh = {}
        self.anomalyQuantile = 0.02

        self.tol_clw = 1e-5

        self.timeStart = 2.5
        self.timeEnd = 3.5

        self.toleranceForInEqualConditions = 0.1

        self.exeDataFolder = self.dataOutputFolder / ( "DATA" + "_" + self.responseVariable)


        FileSystem.makeFolder( self.exeDataFolder )

    def _handleYAML(self):
        self.__readYAML()
        self.__testYAML()

    def __readYAML(self):
        self.yamlDictionary = FileSystem.readYAML(self.inputYAMLFile)
    def __testYAML(self):
        for key in ["responseVariable", "filteringVariablesWithConditions", self.responseIndicatorVariable]:
            assert(key in self.yamlDictionary)

    def testIfResponseVariableIsInFilter(self):
        assert(self.responseVariable in self.filteringVariablesWithConditions)

    def setToleranceForInEqualConditions(self, tolerance):
        self.toleranceForInEqualConditions = tolerance

    def prepare(self):
        self.__prepare__getPhase01()

        self.__prepare__getDesignVariableNames()

        self.__prepare__getResponseFromTrainingSimulations()

        self.__prepare__setSimulationCompleteDataFromDesignAndTraining()

        self.__prepare__setResponseIndicatorFilteredData()

        self.__prepare__getSimulationCollection()

        self.__prepare__filterGetOutlierDataFromLESoutput()

        self.__prepare__filter()

        self.__prepare__saveFilteredDataExeMode()

        self.__prepare__saveFilteredDataRMode()

        self.__prepare__saveOmittedData()

        self.__prepare__getExeFolderList()

    def runEmulator(self):

        if self.useFortran:
            self.__runFortranEmulator()
        else:
            self.__runPythonEmulator()


    def __runFortranEmulator(self):

        self.__fortranEmulator__saveNamelists()

        self.__fortranEmulator__linkExecutables()

        self.__fortranEmulator__runTrain()

        self.__fortranEmulator__runPredictionParallel()

    def __runPythonEmulator(self):

        self.__pythonEmulator__saveYAMLconfigFiles()

        self.__pythonEmulator__linkExecutables()


    def postProcess(self):

        self.collectSimulatedVSPredictedData()

        self._fillUpDrflxValues()

        # self._getWeightedUpdraft()

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

    def __prepare__setResponseIndicatorFilteredData(self):
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



    def collectSimulatedVSPredictedData(self):
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

        dataframe["leaveOneOutIndex"]  = dataframe["wpos"] != -999

        dataframe = dataframe.loc[dataframe["leaveOneOutIndex"]]

        simulated = dataframe[ self.responseVariable ].values
        emulated  = dataframe[ self.emulatedVariable].values


        slope, intercept, r_value, p_value, std_err = stats.linregress(simulated, emulated)

        rSquared = numpy.power(r_value, 2)

        self.leaveOneOutStats = [slope, intercept, r_value, p_value, std_err, rSquared]


    def _getLinearFit(self):

        dataframe = self.simulationCompleteData

        # dataframe = dataframe.loc[ dataframe[self.responseVariable] != self.filterValue]

        extraFilterCondition =  {}
        #extra conditions
        extraFilterCondition["tpot_inv"] = dataframe["tpot_inv"] > 5
        extraFilterCondition["lwpEndValue"] = dataframe["lwpEndValue"] > 10.
        extraFilterCondition["cfracEndValue"] = dataframe["cfracEndValue"] > 0.9
        extraFilterCondition["prcp"] =  dataframe["prcp"] < 1e-6


        self.simulationCompleteData["linearFitIndex"] = Data.getAllConditions(extraFilterCondition)

        filterOutPoints = False
        if filterOutPoints:
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


def main():

    rootFolderOfEmulatorSets = os.environ["EMULATORDATAROOTFOLDER"]
    rootFolderOfDataOutputs = os.environ["EMULATORPOSTPROSDATAROOTFOLDER"]
    inputYamlFile = os.environ["EMULATORINPUTYAML"]

    ###########
    emulatorSets = {"LVL3Night" : EmulatorData("LVL3Night",
                                             [rootFolderOfEmulatorSets,  "case_emulator_DESIGN_v3.0.0_LES_ECLAIR_branch_ECLAIRv2.0.cray.fast_LVL3_night"],
                                             [rootFolderOfDataOutputs],
                                             inputYamlFile
                                             ),
                "LVL3Day"   :  EmulatorData( "LVL3Day",
                                            [rootFolderOfEmulatorSets, "case_emulator_DESIGN_v3.1.0_LES_ECLAIR_branch_ECLAIRv2.0.cray.fast_LVL3_day"],
                                            [rootFolderOfDataOutputs],
                                            inputYamlFile
                                            ),
                "LVL4Night" :  EmulatorData("LVL4Night",
                                            [rootFolderOfEmulatorSets, "case_emulator_DESIGN_v3.2_LES_ECLAIR_branch_ECLAIRv2.0.cray.fast_LVL4_night"],
                                            [rootFolderOfDataOutputs],
                                            inputYamlFile
                                            ),
                "LVL4Day"   : EmulatorData("LVL4Day",
                                            [rootFolderOfEmulatorSets, "case_emulator_DESIGN_v3.3_LES_ECLAIR_branch_ECLAIRv2.0.cray.fast_LVL4_day"],
                                            [rootFolderOfDataOutputs],
                                            inputYamlFile
                                            )
                }

    for key in emulatorSets:
        emulatorSets[key].prepare()
        emulatorSets[key].runEmulator()
        #emulatorSets[key].postProcess()

if __name__ == "__main__":
    start = time.time()

    main()

    end = time.time()
    print(f"Script completed in {Data.timeDuration(end - start):s}")
