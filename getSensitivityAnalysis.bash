#!/bin/bash

for trainingSet in LVL3Night LVL3Day LVL4Night LVL4Day
do
    folder=${EMULATORPOSTPROSDATAROOTFOLDER}/${trainingSet}

    filteredDataInput=${folder}/${trainingSet}_filtered.csv
    sensitivityAnalysisOutput=${folder}/${trainingSet}_sensitivityAnalysis.csv

    Rscript sensitivityAnalysis.r $filteredDataInput $sensitivityAnalysisOutput
    mv Rplots.pdf ${folder}/
done
