from . import halos, fields, visualization, utils, physics, sink, analysis, \
    load_snap, trees

__all__ = ['halos', 'utils', 'trees', 'visualization',
           'fields', 'physics', 'sink', 'analysis']

import yt.units as constants

def load(files='', 
         haloID=None, Galaxy=False, bhID=None, 
         radius=None, bbox=None, 
         MatchObjects=False, fvir=[0.1,0.05,0.5],
         stars=False, dm=False):
    """
    Load a RAMSES output
    CARE : fields depend on the version of ramses
    * files: output/info from ramses, can be the ID of the output. If -1, load the last output

    * stars (False): if True, then add a filter to select star particles
    * dm (False): if True, then add a filter to select dm particles
    
    * MatchObjects: match galaxies and sinks to halos and sinks to galaxies
    * fvir is determines the fraction of the virial radii when matching:
        fvir[0] -> galaxies to halos
        fvir[1] -> sinks to halos
        fvir[2] -> sinks to galaxies

    * bbox: can be used to load a partial data set
    * radius: in the form (10, 'kpc') is the size of the region kept for the dataset
    
    * haloID/bhID : the ID of the halo (or galaxy if Galaxy)/ BH you want to center the box
    """

    ds = load_snap.load(files=files, 
                        haloID=haloID, Galaxy=Galaxy, bhID=bhID, 
                        MatchObjects=MatchObjects, fvir = fvir,
                        radius=radius, bbox=bbox, 
                        stars=stars, dm=dm)

    return ds
