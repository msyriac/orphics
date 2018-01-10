
import numpy as np
import sys
from enlib import enmap,powspec
from orphics.tools.stats import timeit
import orphics.tools.io as io
import subprocess
import logging

class SpectrumVerification(object):
    """
    This class takes in an MPIStats mpibox. It then allows you to take 
    power spectra (with special care for spin-2 maps) and add it to the MPI box.
    The spin-2 rotation matrices and binning factors are pre-calculated. 
    Once you're done, you can make verification plots.

    Each SpectrumVerification object is meant for a specific pixelization.
    However, TODO: overload plus operator to create new SV object from
    two SV objects, which just concatenates the keys, allowing comparisons
    across map pixelizations.
    """

    def __init__(self,mpibox,theory,shape,wcs,lbinner=None,bin_edges=None,pol=False,iau_convention=False):

        self.mpibox = mpibox
        self.theory = theory
        self.shape = shape
        self.pol = pol
        self.wcs = wcs
        self.modlmap = enmap.modlmap(shape,wcs)
        self.fcalc = enmap.FourierCalc(shape,wcs,iau_convention=iau_convention)
        if lbinner is None:
            assert bin_edges is not None
            import orphics.tools.stats as stats
            self.lbinner = stats.bin2D(self.modlmap,bin_edges)
        else:
            self.lbinner = lbinner


    def add_power(self,key,imap,imap2=None,norm=1.,twod_stack=False):
        p2d_all,lteb,lteb2 = self.fcalc.power2d(imap,imap2)
        p2d_all = p2d_all/norm
        if twod_stack: self.mpibox.add_to_stack(key+"_p2d",p2d_all)
        if self.pol:
            clist = ['T','E','B']
            for i,m in enumerate(clist):
                slist = clist[i:]
                for n in slist:
                    j = clist.index(n)
                    spec = m+n
                    p2d = p2d_all[i,j]
                    cents,p1d = self.lbinner.bin(p2d)
                    self.mpibox.add_to_stats(key+spec,p1d)
        else:
            cents,p1d = self.lbinner.bin(p2d_all)
            self.mpibox.add_to_stats(key,p1d)
        self.cents = cents
        return lteb,lteb2

    def plot(self,spec,keys,out_dir=None,scaleY='log',scaleX='log',scale_spectrum=True,xlim=None,ylim=None,skip_uzero=True,pl=None,skip_labels=True):

        
        if pl is None: pl = io.Plotter(scaleY=scaleY,scaleX=scaleX)
        if scale_spectrum:
            scalefac = self.cents**2.
        else:
            scalefac = 1.
        cmb_specs = ['TT','EE','BB','TE','ET','EB','BE','TB','BT']
        uspecs = ['BB','EB','BE','TB','BT']
        done_spec = []

        suffspec = ""
        if (spec in cmb_specs) and self.pol:
            suffspec = spec
        
        for key in keys:

            if ("unlensed" in key) and (spec in uspecs) and skip_uzero: continue

            st = self.mpibox.stats[key+suffspec]
            
            if ("unlensed" in key) or ("delensed" in key):
                spec_key = "u"+spec
            else:
                if (spec in cmb_specs):
                    spec_key = "l"+spec
                else:
                    spec_key = spec

            if spec_key not in done_spec:
                done_spec.append(spec_key)
                th2d = self.theory.gCl(spec_key,self.modlmap)
                cents, th1d = self.lbinner.bin(th2d)
                pl.add(cents,th1d*scalefac)

            pl.addErr(cents,st['mean']*scalefac,yerr=st['errmean']*scalefac,marker="x",ls="none",label=key)

        if xlim is not None: pl._ax.set_xlim(xlim[0],xlim[1])
        if ylim is not None: pl._ax.set_ylim(ylim[0],ylim[1])
        if not(skip_labels): pl.legendOn(labsize=10)
        if pl is None: pl.done(out_dir)
            
        

            
    def plot_diff(self,spec,keys,out_dir=None,scaleY='linear',scaleX='linear',xlim=None,ylim=None,pl=None,skip_labels=True,ratio=True,save_root=None):
        if pl is None: pl = io.Plotter(scaleY=scaleY,scaleX=scaleX)
        cmb_specs = ['TT','EE','BB','TE','ET','EB','BE','TB','BT']
        done_spec = []

        suffspec = ""
        if (spec in cmb_specs) and self.pol:
            suffspec = spec
        
        for key in keys:

            st = self.mpibox.stats[key+suffspec]
            
            if ("unlensed" in key) or ("delensed" in key):
                spec_key = "u"+spec
            else:
                if (spec in cmb_specs):
                    spec_key = "l"+spec
                else:
                    spec_key = spec

            if spec_key not in done_spec:
                done_spec.append(spec_key)
                th2d = self.theory.gCl(spec_key,self.modlmap)
                if ("unlensed" in key) or ("delensed" in key):
                    cents, th1d_unlensed = self.lbinner.bin(th2d)
                else:
                    cents, th1d = self.lbinner.bin(th2d)
                    
            if ("unlensed" in key) or ("delensed" in key):
                th1dnow = th1d_unlensed
            else:
                th1dnow = th1d

                
            rdiff = (st['mean']-th1dnow)
            rerr = st['errmean']
            div = th1dnow if ratio else 1.

            pl.addErr(cents,rdiff/div,yerr=rerr/div,marker="x",ls="none",label=key)
            if save_root is not None: io.save_cols(save_root+spec+"_"+key+".txt",(cents,rdiff/div,rerr/div))

        if not(skip_labels): pl.legendOn(labsize=10)
        if xlim is not None: pl._ax.set_xlim(xlim[0],xlim[1])
        if ylim is not None: pl._ax.set_ylim(ylim[0],ylim[1])
        pl.hline()
        if pl is None: pl.done(out_dir)
        

        


def is_only_one_not_none(a):
    """ Useful for function arguments, returns True if the list 'a'
    contains only one non-None object and False if otherwise.

    Examples:

    >>> is_only_one_not_none([None,None,None])
    >>> False
    >>> is_only_one_not_none([None,1,1,4])
    >>> False
    >>> is_only_one_not_none([1,None,None,None])
    >>> True
    >>> is_only_one_not_none([None,None,6,None])
    >>> True
    """
    return True if count_not_nones(a)==1 else False

def count_not_nones(a):
    return sum([int(x is not None) for x in a])


