__all__ = ['halos', 'utils', 'trees', 'visualization',
           'fields', 'physics', 'snaplist', 'sink', 'analysis']

import yt
from yt.utilities.logger import ytLogger as mylog
import numpy as np
import pandas as pd
import os as os

from . import halos, fields, visualization, utils, physics, sink, analysis, load_snap, galaxies


def load(files='', stars=False, dm=False, MatchObjects=False, bbox=None, haloID=None):
    """
    Load a RAMSES output, with options to filter stars, DM, BHs, or
    halos (from HaloFinder)
    * files: output/info from ramses
    * stars (False): if True, then add a filter to select star particles
    * dm (False): if True, then add a filter to select dm particles
    * bh (False): if True, load BHs
    * halo (False): if True, load halos, tree_brick must be in ./Halos/ID output/tree_brick and
      computed with HaloFinder
    """
    
    ds = load_snap.load(files=files, stars=stars, dm=dm, MatchObjects=MatchObjects, bbox=bbox, haloID=haloID)
     
    return ds
