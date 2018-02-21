#epidistortion test
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import nkepidistortion.nkepitools as nktools 
import numpy.testing as nptest

def test_getArray ():
	assert (nktools.getArray(15,3,5) == ([0,1,2,3,4],[5,6,7,8,9],[10,11,12,13,14])).all()

def test_getArray_numpy ():
	nptest.assert_equal(nktools.getArray(15,3,5),([0,1,2,3,4],[5,6,7,8,9],[10,11,12,13,14]))
	    
