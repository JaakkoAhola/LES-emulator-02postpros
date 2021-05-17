#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 12:23:51 2021

@author: Jaakko Ahola, Finnish Meteorological Institute
@licence: MIT licence Copyright
"""
print(__doc__)
import time
import sys
from EmulatorData import EmulatorData

def main():

    try:
        locationsFile = sys.argv[1]
    except IndexError:
        locationsFile = "/home/aholaj/mounttauskansiot/puhtiwork/EmulatorManuscriptData/locations_local_puhti_mounted.yaml"
        


    ###########
    emulatorSets = {"LVL4Day"   : EmulatorData("LVL4Day",
                                            "case_emulator_DESIGN_v3.3_LES_ECLAIR_branch_ECLAIRv2.0.cray.fast_LVL4_day",
                                            locationsFile
                                            )
                }

    for key in emulatorSets:
        emulatorSets[key].prepare()
        emulatorSets[key].runMethodAnalysis()
        emulatorSets[key].featureImportance()
        emulatorSets[key].bootStrap()
        emulatorSets[key].postProcess()
if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"\nScript completed in { end - start : .1f} seconds")
