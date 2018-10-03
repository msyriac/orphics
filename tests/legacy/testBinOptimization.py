
import numpy as np
import orphics.tools.io as io
import matplotlib.pyplot as plt
import sys 
from orphics.tools.catalogs import split_samples, optimize_splits


"""
This script shows you how to optimize bin edges to get equal S/N.
The example used is a fake data set composed of measurements of 
richness that follow an exponential distribution.

"""

# min and max richness
richness_bounds = [5,100]
a,b = richness_bounds
np.random.seed(100)

# probability function is assumed to be exponential
x0 = 10.  # the knee of the exponential
norm = 6.065 # this norm was explicitly calculated for x0=10 and richness bounds [5,10]
prob = lambda x: np.exp(-x/x0)/norm # analytical pdf of exponential distribution
cdf = lambda x : np.piecewise(x,[x<=a,np.logical_and(x>a,x<b),x>=b],[lambda x: 0.,lambda x: x0*(np.exp(-a/x0)-np.exp(-x/x0))/norm,lambda x:1.]) # analytic cdf of the same

# check cdf and norm
prange = np.linspace(a,b,100)
pl = io.Plotter(labelX="richness $x$",labelY="$P(X<x)$")
pl.add(prange,cdf(prange))
pl.done("cdf.png")
print(("Norm: ", np.trapz(prob(prange),prange)))


# sample using the inverse transform sampling method
inv_cdf = lambda y: -x0*np.log(np.exp(-a/x0)-y*norm/x0) # analytic inverse of cdf
Nsamples = 8000
samples = [inv_cdf(np.random.random()) for x in range(Nsamples)]


# histogram
nbins = 20
n, bins, patches = plt.hist(samples, nbins, facecolor='blue', alpha=0.5,normed=True)
plt.plot(prange,prob(prange))
plt.xlabel("richness $X$")
plt.ylabel("P(X)")





# start with percentile splitting
p = 4  # 4 bins
psplit = 100./p
plist = [psplit*i for i in range(1,p)]
pn = np.percentile(samples,plist).tolist()
in_split_pts = [a]+pn+[b] # the bin edges we start with

# calculate the S/N in each bin
sns,means,Ns = split_samples(samples,in_split_pts)
print ("=== Percentile splitting ===")
print(("S/N: ", sns))
print(("Avg. richness: ", means))
print(("N:", Ns))


 
# === OPTIMIZATION ====        
split_pts = optimize_splits(samples,in_split_pts)
print(("Recommended splitting : ", split_pts))
# =====================

for split_pt in split_pts:
    plt.axvline(x=split_pt,color="k")

s_mean = (np.mean(samples))
s_median = (np.median(samples))
    
plt.axvline(x=s_mean,ls="--",color="C2")
plt.axvline(x=s_median,ls="--",color="C3")

    

plt.savefig("prob.png")
        

# calculate the S/N in optimized binning
sns,means,Ns = split_samples(samples,split_pts)
print ("=== Optimized splitting ===")
print(("S/N: ", sns))
print(("Avg. richness: ", means))
print(("N:", Ns))
