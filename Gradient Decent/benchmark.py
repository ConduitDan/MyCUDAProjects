#!/usr/bin/env python3

import os
import re
import sys
def run(command, file):
    out = -1

    print(command + ' ' + file)
    # Create a temporary file in the same directory
    tmp = 'tmp.out'

    # Run the test
    exec = '( /usr/bin/time ' + command + ' ' + file + ' 1> /dev/null ) 2> ' + tmp
    os.system(exec)

    # If we produced output
    if os.path.exists(tmp):
        out=getoutput(tmp)

        # Delete the temporary file
        #os.system('rm ' + tmp)

    return out
def getoutput(filepath):
    # Load the file
    file_object = open(filepath, 'r')
    lines = file_object.readlines()
    file_object.close()
    # Extract the timing numbers [minutes, seconds]
    times = re.findall(r"[-+]?\d*\.\d+|\d+", lines[0])

    return float(times[0])

    return -1

# do a comparison for 9 levels of cube refinement

# print(run("morpho5","cubeRelaxer.morpho"))
with open('benchmarksAtomic2.txt', 'w') as f:
    original_stdout = sys.stdout
    Names = ["CUDA","CUDAAtomic2","cuBLAS","cuBLASAtomic2"]
    for k in [3]:
        for i in range(7):
            for j in range (3):
                print("opt"+Names[k] + " run " + str(j)+ " Mesh "+str(i+1))
                #print("morpho "+ " run " + str(j)+ " Mesh "+str(i+1))
                sys.stdout = f # Change the standard output to the file we created.
                #print(run("morpho5"," Meshs/cubeRelaxer.morpho"))
                print(run("./opt" + Names[k] + ".out", "Meshs/cube" + str(i+1) +".mesh 10000"))
                sys.stdout = original_stdout # Reset the standard output to its original value









