# -*- coding: utf-8 -*-

from .econtools import *
from .generic import *
from .plots import *

__version__ = '0.1.22'

try:
    import chaospy
except ModuleNotFoundError:
    chaospy = None

if chaospy is not None:
    from .optimize import *

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

if pd is not None:
    from .datatools import *
