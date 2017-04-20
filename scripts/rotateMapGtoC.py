from orphics.tools.curvedSky import slowRotatorGtoC
import healpy as hp

import sys

inFile = sys.argv[1]
outFile = sys.argv[2]



randHp = hp.read_map(inFile)
retMap = slowRotatorGtoC(randHp,hp.get_nside(randHp))
hp.write_map(outFile,retMap)
