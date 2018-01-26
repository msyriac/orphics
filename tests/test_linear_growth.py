import camb
import numpy as np
from orphics import io, cosmology

cc = cosmology.Cosmology(skipCls=True,skipPower=True,skip_growth=False)

z = 0.5

ks= np.logspace(np.log10(0.0001),np.log10(1.0),100)
print(ks)
pl = io.Plotter(xscale='log',ylabel="$f$",xlabel="$k$")

#comps = ['delta_baryon','delta_tot']#,'v_newtonian_cdm','v_baryon_cdm','v_newtonian_baryon']
comps = ['v_newtonian_cdm','v_baryon_cdm','v_newtonian_baryon']
for comp in comps:
    gcomp = cc.growth_scale_dependent(ks,z,comp)
    pl.add(ks,gcomp[:,0,0],label=comp)


pl._ax.axhline(y=cc.fsfunc(cc.z2a(z)),ls="--")
    
pl.legend()
pl.done("output/delta_velocity.png")
