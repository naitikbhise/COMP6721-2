# -------------------------------------------------------
# Project #2 Hacker News Dataset Analysis
# Written by Naitik Bhise (40106507) and Paras Kapoor (40114178)
# For COMP 6721 Section FI – Fall 2019
# --------------------------------------------------------

import pandas as pd

def addDoubleSpacetoFile(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    lines = [line.replace(' ', '  ') for line in lines]
    
    with open(filename, 'w') as f:
        f.writelines(lines) 
        
def writeDataframe(df,filename):
    df.to_csv(filename, header = None, index = True, sep = ' ', mode = 'w')
    addDoubleSpacetoFile(filename)
        
def writeModel(df,filename,AllClasses,appendClassPrefix):
    orderedColumns = []
    for word in AllClasses:
        orderedColumns.append(word)
        orderedColumns.append(appendClassPrefix+word)
    write_df = df.copy()
    write_df = write_df.transpose()
    write_df.index.name = 'TokenName'
    write_df = write_df.reset_index()
    write_df = write_df.sort_values(by ='TokenName')
    write_df = write_df.reset_index()    
    cols = ['TokenName'] + orderedColumns
    write_df = write_df[cols]
    write_df.index += 1
    writeDataframe(write_df,filename)

def WriteList(my_list,filename):
    with open(filename, 'w') as f:
        for item in my_list:
            f.write("%s\n" % item)