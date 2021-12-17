#!/usr/bin/env python3

import os
import re

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

print(run("morpho5","cubeRelaxer.morpho"))

#for i in range(6):
#    print(run("./opt.out", "cube" + str(i+1) +".mesh 10000"))







