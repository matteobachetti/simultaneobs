import os
import pytest
import numpy as np
from astropy.table import QTable
from simultaneobs.simultaneobs import main, summary


class TestFull(object):

    @classmethod
    def setup_class(cls):
        main(['nustar', 'nicer', '--ignore-cache'])
        main(['nustar', 'nicer', '--mjdstart', '55000', '--mjdstop', '58000',
              '--ignore-cache'])
        cls.products = ['nustar-nicer.hdf5',
                        'nustar-nicer_gt55000_lt58000.hdf5']
        cls.outpage = 'outpage.rst'

    @pytest.mark.remote_data
    def test_full_twomissions(self):
        assert os.path.exists('nustar-nicer.hdf5')

    @pytest.mark.remote_data
    def test_full_twomissions_mjdfilt(self):
        main(['nustar', 'nicer', '--mjdstart', '55000', '--mjdstop', '58000',
              '--ignore-cache'])
        outfile = self.products[1]
        assert os.path.exists(outfile)
        table = QTable.read(outfile)
        assert np.any('10302020002' == table['nustar obsid'])

    @pytest.mark.remote_data
    def test_full_twomissions_summary(self):
        summary([self.products[0]])
        assert os.path.exists(self.outpage)

    @classmethod
    def teardown_class(cls):
        import glob
        for fname in cls.products + [cls.outpage]:
            if os.path.exists(fname):
                os.unlink(fname)

        for fname in glob.glob('_*cache.hdf5'):
            os.unlink(fname)
        for fname in glob.glob('_timeline*.hdf5'):
            os.unlink(fname)
