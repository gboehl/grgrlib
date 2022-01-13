#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


def sdw_parser(x):
    """Tries to encode the date format used by the ECB statistical data warehouse (SDW). To be used for the argument `date_parser` in pandas `read_csv`"""

    try:
        return pd.to_datetime(x)
    except:
        pass

    try:
        return pd.to_datetime(x+'0', format='%GW%V%w')
    except:
        pass

    try:
        return pd.to_datetime(x, format='%Y%b')
    except:
        pass

    raise ValueError('Could not find a format for %s.' %x)
