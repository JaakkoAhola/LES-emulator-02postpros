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
import math
import multiprocessing
import numpy
import xarray
import pandas
import pathlib
import time
import os
import re
import sys
from scipy import stats
import LES2emu
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance




sys.path.append(os.environ["LESMAINSCRIPTS"])

from Data import Data
from InputSimulation import InputSimulation

from PostProcessingMetaData import PostProcessingMetaData

sys.path.append(os.environ["PYTHONEMULATOR"])
from GaussianEmulator import GaussianEmulator

class EmulatorData(PostProcessingMetaData):

    def __init__(self, name : str,
                 trainingSimulationRootFolder : list,
                 dataOutputRootFolder : list,
                 configFile : str
                 ):
        
        super().__init__(name, trainingSimulationRootFolder, dataOutputRootFolder, configFile  = configFile)
        
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
        start = time.time()
        print(f"{self.name} Preparing")
        if self.completeFile.is_file():
            # File exists, lets override or read the file
            
            if self.override:
                print(f"{self.name} File exists, but let's override")
                self._prepare_override()
            else:
                print(f"{self.name} File exists, let's read it")
                self.simulationCompleteData = pandas.read_csv( self.completeFile, index_col = 0)
                reg = re.compile("\d{1,9}\S_\d{0,9}")
                if all([reg.match(ind) is not None for ind in self.simulationCompleteData.index.values]):
                    self.simulationCompleteData.index.name = "ID"
                if self.filterIndex in self.simulationCompleteData.columns:
                    self.__prepare_CompleteDataPreExisted()
                else:
                    self.__prepare_appendNewFilterIndexToCompleteData()
            
        else:
            print(f"{self.name} File does not exist, let's create it")
            self._prepare_override()
        
        end = time.time()
        print(f"{self.name} Preparing completed in { end - start : .1f} seconds")
    def __prepare_CompleteDataPreExisted(self):
        self.__prepare__filter()
        self.__prepare__getSimulationCollection()
            
    def __prepare_appendNewFilterIndexToCompleteData(self):
        self.__prepare__setResponseIndicatorCompleteData()
        
        self.__prepare__getSimulationCollection()
        
        self.__prepare__filterGetOutlierDataFromLESoutput()
        
        self.__prepare__filter()
        
        self._fillUpDrflxValues()

        self.__prepare__saveFilteredDataRMode()

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

        self.__prepare__saveFilteredDataRMode()
        
        self.finalise()
        
    def runMethodAnalysis(self):
        start = time.time()
        print(f"{self.name} Method analysis")
        self.__runCrossValidation()
        
        self.finalise()
        end = time.time()
        print(f"{self.name} Method analysis completed in { end - start : .1f} seconds")
    
    def __runCrossValidation(self):
        self.__runCrossValidation_init()
        self.__runCrossValidationLinear()
        self.__runCrossValidationComplexModels()
        self.__crossValidationsToDataFrame()
    
    def __runCrossValidation_init(self):
        self.kfold = KFold(n_splits = self.kFlodSplits, shuffle = True, random_state = 0)
        
        self.inputVariableMatrix = self.simulationFilteredData[ self.designVariableNames ]
        self.responseVariableMatrix = self.simulationFilteredData[ self.responseVariable ]
        self.radiativeWarming = self.simulationCompleteData[self.simulationCompleteData[ self.filterIndex ]][ "drflx" ]
        
        
        outputShape = numpy.shape(self.responseVariableMatrix.values)
        
        self.models ={}
        self.predictions = {}
        for key in self.predictionVariableList:
            
            self.models[key] = [None]*self.kFlodSplits
            self.predictions[key] = numpy.empty(outputShape)
        
        
    def __runCrossValidationLinear(self):
        
        k = 0
        for trainIndex, testIndex in self.kfold.split(self.radiativeWarming):
            
            inputTest = self.__get2DMatrixValues( self.radiativeWarming, testIndex )
            radWarmingTrain = self.__get2DMatrixValues( self.radiativeWarming, trainIndex )
            responseTrain = self.__get2DMatrixValues( self.responseVariableMatrix, trainIndex )
            
            
            linearModel = self._getLinearRegression(radWarmingTrain, responseTrain)
            self.models[self.linearFitVariable][k] = linearModel
            self.predictions[self.linearFitVariable][testIndex] = self._getLinearPredictions(linearModel, inputTest)
            
            k += 1
        
        
    
    def __runCrossValidationComplexModels(self):
        
        k = 0
        for trainIndex, testIndex in self.kfold.split(self.inputVariableMatrix):
            
            inputTrain = self.__get2DMatrixValues( self.inputVariableMatrix, trainIndex )
            inputTest = self.__get2DMatrixValues( self.inputVariableMatrix, testIndex )
            
            
            linearTrain = self.predictions[self.linearFitVariable][trainIndex].reshape(-1,1)
            linearTest = self.predictions[self.linearFitVariable][testIndex].reshape(-1,1)
            
            responseTrain = self.__get2DMatrixValues( self.responseVariableMatrix, trainIndex )
            
            targetCorrectedTrain = responseTrain - linearTrain
            
            
            clfCorrectionModel = self._getRandomForestLinearFitCorrection( \
                                                        {"train" : inputTrain, 
                                                         "linearTrain" : linearTrain},
                                                         {"response" : responseTrain, 
                                                          "corrected" : targetCorrectedTrain})
                
            self.models[self.correctedLinearFitVariable][k] = clfCorrectionModel
            self.predictions[self.correctedLinearFitVariable][testIndex] = self._getRandomForestPredictions(clfCorrectionModel,
                                                         {"input" : inputTest,
                                                          "linear" : linearTest})
                
            emulatorObj = self._getEmulator( inputTrain, responseTrain )
            self.models[self.emulatedVariable][k] = emulatorObj
            self.predictions[self.emulatedVariable][testIndex] = self._getEmulatorPredictions(emulatorObj, inputTest)
            
            k += 1
            
       
    def __crossValidationsToDataFrame(self):
        
        for key in self.predictions:
            self.simulationFilteredData[ key ] = self.predictions[key]
            self.simulationCompleteData.loc[self.filterMask, key] = self.simulationFilteredData[ key ]
        
             
    def __get2DMatrixValues(self, dataframe, indexValues):
        matrix = dataframe.iloc[indexValues].values
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1,1)
        return  matrix
    

    def _getLinearRegression(self, inputs, target):
        linearModel = LinearRegression().fit( inputs, target)
        
        return linearModel
    
    def _getLinearPredictions(self, linearModel, tests ):
        predictions = linearModel.predict( tests )
        predictions = predictions.ravel()
            
        return predictions
    
    def _getRandomForestLinearFitCorrection(self, inputs : dict, targets : dict):
        clfCorrectionModel = RandomForestRegressor(n_estimators=200, n_jobs=-1).fit(numpy.hstack((inputs["train"], inputs["linearTrain"])),
                                                                                            targets["corrected"].ravel())
        return clfCorrectionModel
    
    def _getRandomForestPredictions(self, clfCorrectionModel, tests):
        correction = clfCorrectionModel.predict( self._getRandomForestInput(tests) ).reshape(-1,1)
        predictions = tests["linear"] + correction
        
        predictions = predictions.ravel()
        return predictions
    
    def _getRandomForestInput(self, tests):
        matrix =  numpy.hstack((tests["input"], tests["linear"]))
        return matrix
    
    def _getEmulator(self, inputs, target):
        emulatorObj = GaussianEmulator( inputs, target,
                                    maxiter = self.optimization["maxiter"],
                                    n_restarts_optimizer = self.optimization["n_restarts_optimizer"],
                                    boundOrdo = self.boundOrdo)
        emulatorObj.main()
        
        return emulatorObj
        
        
    def _getEmulatorPredictions(self, emulatorObj, tests):
        emulatorObj.predictEmulator( tests )
        predictions = emulatorObj.getPredictions().ravel()
        
        return predictions
    
    def _getEmulatorInput(self, emulatorObj, tests):
        return emulatorObj.getScaledPredictionMatrix( tests )
    
    def _getEmulatorScaledTarget(self, emulatorObj, target):
        return emulatorObj.getScaledTargetMatrix( target )
    
    def featureImportance(self):
        start = time.time()
        print(f"{self.name} Feature importance")
        self.__collectFeatureImportance()
        self.__featureImportanceToDataframes()
        end = time.time()
        print(f"{self.name} Feature importance completed in { end - start : .1f} seconds")
                                                                               
    def __collectFeatureImportance(self):
        self.permutations = {}
        self.permutations[self.emulatedVariable] = [None]*self.kFlodSplits
        self.permutations[self.correctedLinearFitVariable] = [None]*self.kFlodSplits
        k = 0
        for trainIndex, testIndex in self.kfold.split(self.inputVariableMatrix):
            inputTest = self.__get2DMatrixValues( self.inputVariableMatrix, testIndex )
            linearTest = self.predictions[self.linearFitVariable][testIndex].reshape(-1,1)
            responseTest = self.__get2DMatrixValues( self.responseVariableMatrix, testIndex )
            
            targetCorrectedTest = responseTest - linearTest
            
            emulatorObj = self.models[self.emulatedVariable][k]
            emulatorModel = emulatorObj.getEmulator()
            
            emulatorFeatImportance = permutation_importance( emulatorModel,
                                                                      self._getEmulatorInput(emulatorObj, inputTest),
                                                                      self._getEmulatorScaledTarget(emulatorObj, responseTest) )["importances_mean"]
            self.permutations[self.emulatedVariable][k] = emulatorFeatImportance
            self.permutations[self.correctedLinearFitVariable][k] = permutation_importance(self.models[self.correctedLinearFitVariable][k], 
                                                                                self._getRandomForestInput({"input": inputTest,
                                                                                                            "linear" : linearTest})
                                                                                , targetCorrectedTest)["importances_mean"]
            k +=1
    
    def __featureImportanceToDataframes(self):
        self.__featureVariables()
        self.__featureData()
        self.__featureDataFrame()
        self.__featureFinalise()
        
    def __featureVariables(self):
        self.featureVariables = self.designVariableNames + [self.responseVariable + "_linearFit"]
        self.featureMeanVariables = [kk + "_FeatureImportanceMean" for kk in self.featureVariables]
        self.featureStdVariables = [kk + "_FeatureImportanceStd" for kk in self.featureVariables]
        self.featureColumns = self.featureMeanVariables + self.featureStdVariables
    
    def __featureData(self):
        data ={}
        data[self.emulatedVariable] = {}
        data[self.correctedLinearFitVariable] = {}
        
        for key in data:
            data[key]["mean"] = numpy.asarray(self.permutations[key][:]).mean(axis = 0)
            data[key]["std"] = numpy.asarray(self.permutations[key][:]).std(axis = 0)
        
        for subkey in data[self.emulatedVariable]:
            data[self.emulatedVariable][subkey] = numpy.append(data[self.emulatedVariable][subkey], numpy.nan)
            
        self.featuredataframedata = {}
        
        for key in data:
            self.featuredataframedata[key] = numpy.concatenate( (data[key]["mean"], data[key]["std"]))
    def __featureDataFrame(self):
        self.featureDataFrame = pandas.DataFrame.from_dict(self.featuredataframedata, orient="index", columns = self.featureColumns)
        
    def __featureFinalise(self):
        self.featureDataFrame.to_csv(self.featureImportanceFile)
    
    def bootStrap(self):
        if self.runBootStrap:
            start = time.time()
            print(f"{self.name} Bootstrapping")
            
            self.__init_bootstrap()
            self.__bootstrap_linear()
            self.__bootstrap_complexModels()
            self.__bootstrap_metricsByIteration()
            self.__bootstrap_StatsCombined()
            self.__bootstrap_dataframe()
            self.__bootstrap_finalise()
        
            end = time.time()
            print(f"{self.name} Bootstrapping completed in { end - start : .1f} seconds")
        

    def postProcess(self):
        
        if self.runPostProcessing:
            start = time.time()
            print(f"{self.name} Postprocessing")
            self._runPostProcess()
            end = time.time()
            print(f"{self.name} Postprocessing completed in { end - start : .1f} seconds")
        else:
            print(f"{self.name} Not postprocessing because runPostProcess set to False in configFile")
        
    def _runPostProcess(self):    
        
        self._getAnomalyLimitsQuantile()
        self._getAnomalyLimitsConstant()

        self._getAnomalies()

        self._getStats()

        self._finaliseStats()
        self._finaliseAnomalies()

        self.finalise()



    def __prepare__getPhase01(self):
        assert(self.phase01CSVFile.is_file())
        self.phase01 = pandas.read_csv(self.phase01CSVFile, index_col=0)

    def __prepare__getDesignVariableNames(self):
        

        self.microphysics = self.ID_prefix[0]

        self.timeOfDay = self.ID_prefix[1]

        if self.microphysics == "3":
            self.microphysicsVariables = self.microphysicsVariablesPool["SB"]
        elif self.microphysics == "4":
            self.microphysicsVariables = self.microphysicsVariablesPool["SALSA"]

        if self.timeOfDay == "D":
            self.timeOfDayVariable = self.solarZenithAngle
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
        self.filterMask = self.simulationCompleteData[ self.filterIndex ]
        
        self.simulationCompleteDataMasked = self.simulationCompleteData[ self.filterMask ]
        
        filteredData = self.simulationCompleteDataMasked.copy()

        self.simulationFilteredData = filteredData[self.designVariableNames + [ self.responseVariable ]]
        
        self.simulationFilteredCSV = filteredData[self.designVariableNames + [ self.responseVariable ]]

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

                            
        
    def __init_bootstrap(self):
        self.bootstrapStats = {}
        
        self.bootstrapKFold = KFold(n_splits = self.kFlodSplits, shuffle = True, random_state = 0)
        
        self.bootstrapModelList = ["linearFit", "emulator", "correctedLinearFit", "simulated"]
        
        self.bootstrapMetrics = ["rSquared", "rmse"]
        
        self.bootstrapModelValues = {}
        
        for key in self.bootstrapModelList:
            self.bootstrapModelValues[key] = numpy.empty((self.bootstrappingParameters["iterations"], self.bootstrappingParameters["sampleSize"]))
        
        for key in self.bootstrapModelList[:-1]:
            self.bootstrapStats[key] = {}
            for metric in self.bootstrapMetrics:
                self.bootstrapStats[key][metric] = numpy.empty( self.bootstrappingParameters["iterations"] )
        
        
    def __bootstrap_linear(self):
    
        for bootStrapIndex in range(self.bootstrappingParameters["iterations"]):
            
            sampleDF = self.simulationCompleteData.sample(n = self.bootstrappingParameters["sampleSize"], random_state = bootStrapIndex).sort_index()        
        
            BSresponseVariableMatrix = sampleDF[ self.responseVariable ]
            BSradiativeWarming = sampleDF["drflx"]
            
            self.bootstrapModelValues["simulated"][bootStrapIndex][:] = BSresponseVariableMatrix
            
            for trainIndex, testIndex in self.bootstrapKFold.split(BSradiativeWarming):
                inputTest = self.__get2DMatrixValues( BSradiativeWarming, testIndex )
                radWarmingTrain = self.__get2DMatrixValues( BSradiativeWarming, trainIndex )
                responseTrain = self.__get2DMatrixValues( BSresponseVariableMatrix, trainIndex )
            
            
                linearModel = self._getLinearRegression(radWarmingTrain, responseTrain)
                self.bootstrapModelValues["linearFit"][bootStrapIndex][testIndex] = self._getLinearPredictions(linearModel, inputTest)
    
    def __bootstrap_complexModels(self):
        for bootStrapIndex in range(self.bootstrappingParameters["iterations"]):
            
            sampleDF = self.simulationCompleteData.sample(n = self.bootstrappingParameters["sampleSize"], random_state = bootStrapIndex).sort_index()        
        
            BSinputVariableMatrix =  sampleDF[ self.designVariableNames]
            BSresponseVariableMatrix = sampleDF[ self.responseVariable ]
            
            for trainIndex, testIndex in self.bootstrapKFold.split(BSinputVariableMatrix):
            
                inputTrain = self.__get2DMatrixValues( BSinputVariableMatrix, trainIndex )
                inputTest = self.__get2DMatrixValues( BSinputVariableMatrix, testIndex )
                
                
                linearTrain = self.bootstrapModelValues["linearFit"][bootStrapIndex][trainIndex].reshape(-1,1)
                linearTest = self.bootstrapModelValues["linearFit"][bootStrapIndex][testIndex].reshape(-1,1)
                
                responseTrain = self.__get2DMatrixValues( BSresponseVariableMatrix, trainIndex )
                
                targetCorrectedTrain = responseTrain - linearTrain
                
                
                clfCorrectionModel = self._getRandomForestLinearFitCorrection( \
                                                            {"train" : inputTrain, 
                                                             "linearTrain" : linearTrain},
                                                             {"response" : responseTrain, 
                                                              "corrected" : targetCorrectedTrain})
                    
                self.bootstrapModelValues["correctedLinearFit"][bootStrapIndex][testIndex] = self._getRandomForestPredictions(clfCorrectionModel,
                                                             {"input" : inputTest,
                                                              "linear" : linearTest})
                    
                emulatorObj = self._getEmulator( inputTrain, responseTrain )
                self.bootstrapModelValues["emulator"][bootStrapIndex][testIndex] = self._getEmulatorPredictions(emulatorObj, inputTest)

    def __bootstrap_metricsByIteration(self):
        for key in self.bootstrapModelList[:-1]:
            for bootStrapIndex in range(self.bootstrappingParameters["iterations"]):
                predicted = self.bootstrapModelValues[key][bootStrapIndex]
                true = self.bootstrapModelValues["simulated"][bootStrapIndex]
                
                statistics = self._statsMethod(true, predicted)
                
                for metric in self.bootstrapMetrics:
                    self.bootstrapStats[key][metric][bootStrapIndex] = statistics[metric]

    def __bootstrap_StatsCombined(self):
        self.bootstrapAnalysis = {}                 
        for key in self.bootstrapStats:
            
            self.bootstrapAnalysis[key] = {}
            
            for metric in self.bootstrapStats[key]:
                self.bootstrapAnalysis[key][metric + "_Mean" ] = self.bootstrapStats[key][metric].mean()
                self.bootstrapAnalysis[key][metric + "_Std" ] = self.bootstrapStats[key][metric].std()
    
    def __bootstrap_dataframe(self):
        self.bootstrapDataFrame = pandas.DataFrame.from_dict(self.bootstrapAnalysis, orient="index")

    def __bootstrap_finalise(self):
        self.bootstrapDataFrame.to_csv(self.bootStrapFile)

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

        print(f"{self.name} Start calculation of weighted updrafts " + self.name)

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

        print(f"{self.name} Time to calculate updrafts {t2-t1:.1f}")



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
    
    def _getStats(self):
        
        self.statistics = {}
        
        self._getEmulatorStats()
        self._getLinearFitStats()
        self._getCorrectedLinearFitStats()
        
        self._getLinearFitRadWarminStats()
        

    def _getEmulatorStats(self):

        simulated = self.simulationCompleteDataMasked[ self.responseVariable ].values
        emulated = self.simulationCompleteDataMasked[ self.emulatedVariable ].values

        self.statistics["emulator"] = self._statsMethod(simulated, emulated)
    
    def _getLinearFitStats(self):
        simulated = self.simulationCompleteDataMasked[ self.responseVariable ].values
        linearFit = self.simulationCompleteDataMasked[ self.linearFitVariable ].values
        
        self.statistics["linearFit"] = self._statsMethod( simulated, linearFit)
        

    def _getLinearFitRadWarminStats(self):

        radiativeWarming  = self.simulationCompleteDataMasked["drflx"].values
        updraft =  self.simulationCompleteDataMasked[ self.responseVariable ].values

        self.statistics["linearFitRadWarming"] = self._statsMethod(radiativeWarming, updraft)

        
    def _getCorrectedLinearFitStats(self):
        
        simulated =  self.simulationCompleteDataMasked[ self.responseVariable ].values
        corrected = self.simulationCompleteDataMasked[ self.correctedLinearFitVariable ].values


        self.statistics["correctedLinearFit"] = self._statsMethod(simulated, corrected)

    def _statsMethod(self, observed, predicted):
        slope, intercept, r_value, p_value, std_err = stats.linregress(observed, predicted)

        rSquared = numpy.power(r_value, 2)
        
        rmse = mean_squared_error(observed, predicted, squared=False)
        
        statistics = {"slope":slope, 
                      "intercept": intercept,
                      "r_value":r_value,
                      "p_value" :p_value,
                      "std_err": std_err,
                      "rSquared":rSquared,
                      "rmse":rmse}
        
        return statistics

    def _finaliseStats(self):
        self.statsDataFrame = pandas.DataFrame.from_dict(self.statistics, orient = "index")

        self.statsDataFrame.to_csv( self.statsFile)
    def finalise(self):
        self.simulationCompleteData.to_csv(self.completeFile)
