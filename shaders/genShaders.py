# genShaders.py

import sys
import os
import subprocess

def writeShader(base, name, type, definitions):
    
    inFile = open(base + "." + type + ".glsl", "r")
    outFileName = name + "." + type + ".glsl"

    if (os.path.exists(outFileName)):
      os.remove(outFileName)

    outFile = open(outFileName, "w")
    outFile.write("#version 450\n" + definitions + inFile.read())

    inFile.close()
    outFile.close()

base = sys.argv[1]
name = sys.argv[2]
options =sys.argv[3:]
definitions = ''.join(["#define " + option + "\n" for option in options])

writeShader(base, name, "vert", definitions)
writeShader(base, name, "frag", definitions)

subprocess.run(['glslangValidator', '-V', f'{name}.vert.glsl', '-o', f'{name}.vert.spv', '-S', 'vert'])
subprocess.run(['glslangValidator', '-V', f'{name}.frag.glsl', '-o', f'{name}.frag.spv', '-S', 'frag'])
