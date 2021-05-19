#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from julia import Main as japi
from julia import Flatten as jflatten
from julia.Flatten import reconstruct as restruct

def setfield(obj, field, val):

    japi.tmp = obj
    japi.eval("using Setfield")
    japi.eval("@set! tmp.%s = %s" %(field, float(val)))

    return japi.tmp
