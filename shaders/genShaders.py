# genShaders.py

import sys
import os

def writeShader(name, type, definitions):
    
    inFile = open("base." + type + ".glsl", "r")
    outFileName = name + "." + type + ".glsl"

    if (os.path.exists(outFileName)):
      os.remove(outFileName)

    outFile = open(outFileName, "w")
    outFile.write("#version 450\n" + definitions + inFile.read())

    inFile.close()
    outFile.close()

name = sys.argv[1]
options =sys.argv[2:]
definitions = ''.join(["#define " + option + "\n" for option in options])

writeShader(name, "frag", definitions)
writeShader(name, "vert", definitions)
