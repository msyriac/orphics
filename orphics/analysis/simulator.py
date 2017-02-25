import warnings


class Power(object):

    def __init__(self,templateMap,power2DData=None,ells=None,Cls=None):

        #self.Nx = templateMap.Nx
        #self.Ny = templateMap.Ny
        
        if power2DData is not None:
            if (ells is not None) or (Cls is not None): warnings.warn("Warning: ignoring ells and Cls")
            self.power2d = power2DData.copy()

        else:
            assert ells is not None
            assert Cls is not None




class Simulator(object):
    
    def __init__(self):
        pass





class GaussianSimulator(Simulator):
    

    def __init__(self,Powers):
        pass



