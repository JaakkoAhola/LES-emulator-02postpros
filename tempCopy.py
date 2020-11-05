#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 14:54:58 2020

@author: Jaakko Ahola, Finnish Meteorological Institute
@licence: MIT licence Copyright
"""
import time
import pandas
import pathlib
import os
import copy
import shutil

def checkExistance(name, file):
    prefix = name[3:5]
    file = pathlib.Path(file)
    
    source = list(file.parts)[:6]
    dest = copy.deepcopy(source)
    dest[4] = "EmulatorManuscriptData2"
    dest = pathlib.Path(os.path.join(*dest)) / "DATA_wpos"
    source = pathlib.Path(os.path.join(*source)) / "DATA_wpos"
    
    
    df = pandas.read_csv(file)
    
    
    for i in range(df.shape[0]):
        if i not in df.designCase.values:
            print(prefix + "_" + str(i+1).zfill(3))
    
    # for indeksi, designCase in enumerate(df.designCase.values):
        
        
        # source_subfolder = str(indeksi)
        # dest_subfolder = prefix + "_" + str(designCase+1).zfill(3)
        
        
        
        
        # GP_source_file = source / source_subfolder / "out.gp"
        # GP_dest_file = dest / dest_subfolder / ( "out_" + dest_subfolder + ".gp" )
        
        # predictOutput_source_file = source / source_subfolder / "DATA_predict_output"
        
        # predictOutput_dest_file = dest / dest_subfolder / ( "DATA_predict_" + dest_subfolder )
        
        # if GP_source_file.is_file():
        #     pass
        # else:
        #     print("NOOO")
            
        # if ( GP_dest_file.parent.is_dir()):
        #     pass
        # else:
        #     print("EIIIIII")
            
        # if predictOutput_source_file.is_file():
        #     pass
        # else:
        #     print("NOOO2")
            
        # if ( predictOutput_dest_file.parent.is_dir()):
        #     pass
        # else:
        #     print("EIIIIII2")
        
        
        # shutil.copy(GP_source_file, GP_dest_file )
        # shutil.copy(predictOutput_source_file, predictOutput_dest_file)
            
            
def main():
   dictionary = {"LVL3Day" : "/home/aholaj/Data/EmulatorManuscriptData/LVL3Day/LVL3Day_simulatedVSPredictedData.csv",
                  "LVL3Night" : "/home/aholaj/Data/EmulatorManuscriptData/LVL3Night/LVL3Night_simulatedVSPredictedData.csv",
                  "LVL4Day" : "/home/aholaj/Data/EmulatorManuscriptData/LVL4Day/LVL4Day_simulatedVSPredictedData.csv",
                  "LVL4Night" : "/home/aholaj/Data/EmulatorManuscriptData/LVL4Night/LVL4Night_simulatedVSPredictedData.csv"
                 }
   
   for key in dictionary:
       checkExistance(key, dictionary[key])
if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("Script completed in " + str(round((end - start),0)) + " seconds")
