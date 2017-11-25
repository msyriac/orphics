import yaml
import numpy as np
from enlib import enmap


class ACTPolMapReader(object):

    def __init__(self,config_yaml_path):

        with open(config_yaml_path, 'r') as ymlfile:
            self._cfg = yaml.load(ymlfile,Loader=yaml.BaseLoader)

        self.map_root = self._cfg['map_root']
        self.beam_root = self._cfg['beam_root']

    def patch_bounds(self,patch):
        return (np.array([float(x) for x in self._cfg['patches'][patch].split(',')])*np.pi/180.).reshape((2,2))
        
    def get_beam(self,season,patch,array,freq="150",day_night="night"):
        beam_file = self.beam_root+self._cfg[season][array][freq][patch][day_night]['beam']
        ls,bells = np.loadtxt(beam_file,usecols=[0,1],unpack=True)
        return ls, bells

        
    def get_map(self,split,season,patch,array,freq="150",day_night="night",full_map=False,weight=False,get_identifier=False,t_only=False):

        maps = []
        maplist = ['srcfree_I','Q','U'] if not(t_only) else ['srcfree_I']
        for pol in maplist if not(weight) else [None]:
            fstr = self._hstring(season,patch,array,freq,day_night) if weight else self._fstring(split,season,patch,array,freq,day_night,pol)
            cal = float(self._cfg[season][array][freq][patch][day_night]['cal']) if not(weight) else 1.
            fmap = enmap.read_map(fstr)*np.sqrt(cal) 
            if not(full_map):
                bounds = self.patch_bounds(patch) 
                retval = fmap.submap(bounds)
            else:
                retval = fmap
            maps.append(retval)
        retval = enmap.ndmap(np.stack(maps),maps[0].wcs) if not(weight) else maps[0]

        if get_identifier:
            identifier = '_'.join(map(str,[season,patch,array,freq,day_night]))
            return retval,identifier
        else:
            return retval
        

    def _fstring(self,split,season,patch,array,freq,day_night,pol):
        # Change this function if the map naming scheme changes
        splitstr = "set0123" if split<0 or split>3 else "set"+str(split)
        return self.map_root+season+"/"+patch+"/"+season+"_mr2_"+patch+"_"+array+"_f"+freq+"_"+day_night+"_"+splitstr+"_wpoly_500_"+pol+".fits"

    def _hstring(self,season,patch,array,freq,day_night):
        splitstr = "set0123"
        return self.map_root+season+"/"+patch+"/"+season+"_mr2_"+patch+"_"+array+"_f"+freq+"_"+day_night+"_"+splitstr+"_hits.fits"
