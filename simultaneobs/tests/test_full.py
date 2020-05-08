import os
import pytest
import numpy as np
from astropy.table import QTable
from simultaneobs.simultaneobs import main


@pytest.mark.remote_data
def test_full_twomissions():
    main(['nustar', 'nicer', '--ignore-cache'])
    assert os.path.exists('nustar-nicer.hdf5')


@pytest.mark.remote_data
def test_full_twomissions_mjdfilt():
    main(['nustar', 'nicer', '--mjdstart', '55000', '--mjdstop', '58000',
          '--ignore-cache'])
    outfile = 'nustar-nicer_gt55000_lt58000.hdf5'
    assert os.path.exists(outfile)
    table = QTable.read(outfile)
    assert np.any('10302020002' == table['nustar obsid'])