from astLib import astWCS, astCoords
import liteMap as lm
import time
import sys


def downWriteMap(fname):
    nowMap = lm.liteMapFromFits(fname)
    lowTemp = lm.getEmptyMapWithDifferentDims(nowMap,int(nowMap.Ny/2.),int(nowMap.Nx/2.))
    lowMap = lm.resampleFromHiResMap(nowMap, lowTemp)
    lowMap.writeFits(fname[:-5]+"_down.fits",overWrite=True)
    # try:
    #     lowMap.writeFits(fname[:-5]+"_down.fits")
    # except:
    #     pass


simRoot = "/astro/astronfs01/workarea/msyriac/cmbSims/"


lensedTPath = lambda x: simRoot + "lensedCMBMaps_" + str(x).zfill(5) + "/order5_periodic_lensedCMB_T_beam_0.fits"
lensedQPath = lambda x: simRoot + "lensedCMBMaps_" + str(x).zfill(5) + "/order5_periodic_lensedCMB_Q_beam_0.fits"
lensedUPath = lambda x: simRoot + "lensedCMBMaps_" + str(x).zfill(5) + "/order5_periodic_lensedCMB_U_beam_0.fits"
kappaPath = lambda x: simRoot + "phiMaps_" + str(x).zfill(5) + "/kappaMap_0.fits"


i = int(sys.argv[1])

print(i)

print("Trimming lensed T")
downWriteMap(lensedTPath(i))
print("Trimming lensed Q")
downWriteMap(lensedQPath(i))
print("Trimming lensed U")
downWriteMap(lensedUPath(i))
print("Trimming lensed kappa")
downWriteMap(kappaPath(i))


# N = 100



# for i in range(N):
#     startTime = time.time()


#     downWriteMap(lensedTPath(i))
#     downWriteMap(lensedQPath(i))
#     downWriteMap(lensedUPath(i))
#     downWriteMap(kappaPath(i))

#     print "Done in ", time.time()-startTime, " seconds ."
#     sys.exit()
