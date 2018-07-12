from . import halos, fields, visualization, utils, physics, sink, analysis, \
    load_snap, snaplist, trees

__all__ = ['halos', 'utils', 'trees', 'visualization',
           'fields', 'physics', 'snaplist', 'sink', 'analysis']


def load(files='', MatchObjects=False, bbox=None, haloID=None, Galaxy=False,
         bhID=None, radius=None, stars=False, dm=False):
    """
    Load a RAMSES output
    CARE : fields depend on the version of ramses
    * files: output/info from ramses, can be the ID of the output
    * stars (False): if True, then add a filter to select star particles
    * dm (False): if True, then add a filter to select dm particles
    * MatchObjects: match galaxies and sinks to halos and sinks to galaxies
    * bbox: can be used to load a partial data set
    * haloID/bhID : the ID of the halo (or galaxy if Galaxy)/ BH you want to center the box
    * radius: in the form (10, 'kpc') is the size of the region kept for the dataset
    """

    ds = load_snap.load(files=files, MatchObjects=MatchObjects,
                        bbox=bbox, haloID=haloID, Galaxy=Galaxy,
                        bhID=bhID, radius=radius, stars=stars, dm=dm)

    return ds
