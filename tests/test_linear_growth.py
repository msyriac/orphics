import camb
import numpy as np
from orphics import io, cosmology
import matplotlib.cm as cm

cc = cosmology.Cosmology(skipCls=True,skipPower=True,skip_growth=False)

z = 0.5

ks= np.logspace(np.log10(0.0001),np.log10(1.0),100)
pl = io.Plotter(xscale='log',ylabel="$D$",xlabel="$k$")

#comps = ['delta_baryon','delta_tot']#,'v_newtonian_cdm','v_baryon_cdm','v_newtonian_baryon']
#comps = ['delta_tot']#,'v_newtonian_cdm','v_baryon_cdm','v_newtonian_baryon']
comps = ['growth']
zs = np.arange(0.,3.,0.1)
for z in zs:
    print(z)
    #comps = ['v_newtonian_cdm','v_baryon_cdm','v_newtonian_baryon']
    for comp in comps:
        gcomp = cc.growth_scale_dependent(ks,z,comp)
        pl.add(ks,gcomp[:,0,0],color=cm.hot(z))#,label=comp)


pl._ax.axhline(y=cc.Dfunc(cc.z2a(z)),ls="--")
pl._ax.axhline(y=cc.fsfunc(cc.z2a(z)),ls="--")
    
pl.legend()
#pl.done("output/delta_velocity.png")
pl.done("output/delta_growth_rate.png")
