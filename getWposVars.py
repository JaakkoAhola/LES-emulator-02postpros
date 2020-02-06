#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 17:31:36 2020

@author: Jaakko Ahola, Finnish Meteorological Institute
@licence: MIT licence Copyright
"""
import pathlib
import time

import LES2emu

def saveUpdraftFromTrainingSimulations(emulatorFolder, csvFolder, csvFilename):
    EmuVars = LES2emu.GetEmu2Vars( str(emulatorFolder) )
    absolutePathCSVFile = pathlib.Path(csvFolder) / csvFilename
    print(absolutePathCSVFile)
    with open( absolutePathCSVFile,'w') as file:
        file.writelines(["%s\n" % (item[7])  for item in EmuVars])
    
def main():
   rootFolderOfEmulatorSets = "/home/aholaj/mounttauskansiot/eclairmount"
    
   folderList = {"LVL3Night" :"case_emulator_DESIGN_v3.0.0_LES_ECLAIR_branch_ECLAIRv2.0.cray.fast_LVL3_night",
                "LVL3Day"   :  "case_emulator_DESIGN_v3.1.0_LES_ECLAIR_branch_ECLAIRv2.0.cray.fast_LVL3_day" ,
                "LVL4Night" :  "case_emulator_DESIGN_v3.2_LES_ECLAIR_branch_ECLAIRv2.0.cray.fast_LVL4_night" ,
                "LVL4Day"   : "case_emulator_DESIGN_v3.3_LES_ECLAIR_branch_ECLAIRv2.0.cray.fast_LVL4_day" 
                }
   
   for ind, key in enumerate(list(folderList)):
       folderList[key] = pathlib.Path(rootFolderOfEmulatorSets) / folderList[key]
   csvFolder = "/home/aholaj/OneDrive/000_WORK/000_ARTIKKELIT/001_Manuscript_LES_emulator/data"
   ###########
   csvFolder = pathlib.Path(csvFolder)
   for ind, key in enumerate(list(folderList)):
       saveUpdraftFromTrainingSimulations( folderList[key], csvFolder / str(key) ,  str(key) + "_wpos.csv" )
    
   
if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("Script completed in " + str(round((end - start),0)) + " seconds")
