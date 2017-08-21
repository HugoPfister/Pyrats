#!/usr/bin/env python

"""Module to deal with halos, to be used with HaloMaker.

This module is heavily inspired by the set of IDL routines originally found
in the Ramses Analysis ToolSuite (RATS).

TODO: Some more documentation
"""

import numpy as np
import pandas as pd
import utils
import yt
import glob
import pyrats

class SnapList(object):
    def __init__(self, bh=False, halo=False, stars=False, dm=False):
        """Some documentation, list of useful function, etc."""
        folder=glob.glob('output*/info*')
        folder.sort()
        self.ds=[pyrats.load(f, bh=bh, halo=halo, stars=stars, dm=dm) for f in folder]       


    def get_bh_prop(self, prop='M', time=False):
        data={}
        for ds in self.ds:
            if time:
                t=ds.current_time.in_units('Myr')
                unitt='time'
            else:
                t=ds.current_redshift
                unitt='z'
            for bhid in ds.sink.ID:
                bh=ds.sink[ds.sink.ID==bhid]
                if data.has_key(float(bh.ID)):
                    data[float(bh.ID)][unitt]+=[t]
                    data[float(bh.ID)][prop]+=[float(bh[prop])]
                else:
                    data[float(bh.ID)]={unitt:[t],prop:[float(bh[prop])]}
        return data

