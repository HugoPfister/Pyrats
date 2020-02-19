 #   load_snap, trees
from . import load_snap, visualization

#__all__ = ['utils', 'trees', 'visualization',
#           'fields', 'physics', 'sink', 'analysis']

import yt.units as constants

def load(files='',
         haloID=None, Galaxy=False, bhID=None,
         radius=None, bbox=None,
         MatchObjects=False, fvir=[1,0.1,'r90'], contam=False,
         old_ramses=False, verbose=True, prefix='.'):
    """
    Load a RAMSES output
    See pyrats.load.load_snap
    """

    ds = load_snap.load(files=files,
                        haloID=haloID, Galaxy=Galaxy, bhID=bhID,
                        MatchObjects=MatchObjects, fvir = fvir, contam=contam,
                        radius=radius, bbox=bbox,
                        old_ramses=old_ramses, verbose=verbose, prefix=prefix)

    return ds
