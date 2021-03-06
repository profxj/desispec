import unittest, os
from uuid import uuid1

import numpy as np

from desispec.frame import Frame
from desispec.fiberflat import FiberFlat
from desispec.sky import SkyModel
import desispec.io
from astropy.io import fits
from shutil import rmtree

class TestIO(unittest.TestCase):

    #- Create unique test filename in a subdirectory
    @classmethod
    def setUpClass(cls):
        cls.testfile = 'test-{uuid}/test-{uuid}.fits'.format(uuid=uuid1())
        cls.testDir = os.path.join(os.environ['HOME'],'desi_test_io')
        cls.origEnv = {'PRODNAME':None,
            "DESI_SPECTRO_DATA":None,
            "DESI_SPECTRO_REDUX":None}
        cls.testEnv = {'PRODNAME':'dailytest',
            "DESI_SPECTRO_DATA":os.path.join(cls.testDir,'spectro','data'),
            "DESI_SPECTRO_REDUX":os.path.join(cls.testDir,'spectro','redux')}
        for e in cls.origEnv:
            if e in os.environ:
                cls.origEnv[e] = os.environ[e]
            os.environ[e] = cls.testEnv[e]

    #- Cleanup test files if they exist
    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.testfile):
            os.remove(cls.testfile)
            testpath = os.path.normpath(os.path.dirname(cls.testfile))
            if testpath != '.':
                os.removedirs(testpath)
        for e in cls.origEnv:
            if cls.origEnv[e] is None:
                del os.environ[e]
            else:
                os.environ[e] = cls.origEnv[e]
        if os.path.exists(cls.testDir):
            rmtree(cls.testDir)

    def test_fitsheader(self):
        #- None is ok; just returns blank Header
        header = desispec.io.util.fitsheader(None)
        self.assertTrue(isinstance(header, fits.Header))
        self.assertEqual(len(header), 0)

        #- input is dict
        hdr = dict()
        hdr['BLAT'] = 'foo'
        hdr['BAR'] = (1, 'biz bat')
        header = desispec.io.util.fitsheader(hdr)
        self.assertTrue(isinstance(header, fits.Header))
        self.assertEqual(header['BLAT'], 'foo')
        self.assertEqual(header['BAR'], 1)
        self.assertEqual(header.comments['BAR'], 'biz bat')

        #- input header as a list, get a fits.Header back
        hdr = list()
        hdr.append( ('BLAT', 'foo') )
        hdr.append( ('BAR', (1, 'biz bat')) )
        header = desispec.io.util.fitsheader(hdr)
        self.assertTrue(isinstance(header, fits.Header))
        self.assertEqual(header['BLAT'], 'foo')
        self.assertEqual(header['BAR'], 1)
        self.assertEqual(header.comments['BAR'], 'biz bat')

        #- fits.Header -> fits.Header
        header = desispec.io.util.fitsheader(header)
        self.assertTrue(isinstance(header, fits.Header))
        self.assertEqual(header['BLAT'], 'foo')
        self.assertEqual(header['BAR'], 1)
        self.assertEqual(header.comments['BAR'], 'biz bat')

        #- Can't convert and int into a fits Header
        self.assertRaises(ValueError, desispec.io.util.fitsheader, (1,))

    def test_frame_rw(self):
        nspec, nwave, ndiag = 5, 10, 3
        flux = np.random.uniform(size=(nspec, nwave))
        ivar = np.random.uniform(size=(nspec, nwave))
        mask_int = np.zeros((nspec, nwave), dtype=int)
        mask_uint = np.zeros((nspec, nwave), dtype=np.uint32)
        wave = np.arange(nwave)
        R = np.random.uniform( size=(nspec, ndiag, nwave) )

        for mask in (mask_int, mask_uint):
            frx = Frame(wave, flux, ivar, mask, R)
            desispec.io.write_frame(self.testfile, frx)
            frame = desispec.io.read_frame(self.testfile)

            self.assertTrue(np.all(flux == frame.flux))
            self.assertTrue(np.all(ivar == frame.ivar))
            self.assertTrue(np.all(wave == frame.wave))
            self.assertTrue(np.all(mask == frame.mask))
            self.assertTrue(np.all(R == frame.resolution_data))
            self.assertTrue(frame.resolution_data.dtype.isnative)

    def test_sky_rw(self):
        nspec, nwave = 5,10
        wave = np.arange(nwave)
        flux = np.random.uniform(size=(nspec, nwave))
        ivar = np.random.uniform(size=(nspec, nwave))
        mask_int = np.zeros(shape=(nspec, nwave), dtype=int)
        mask_uint = np.zeros(shape=(nspec, nwave), dtype=np.uint32)

        for mask in (mask_int, mask_uint):
            # skyflux,skyivar,skymask,cskyflux,cskyivar,wave
            sky = SkyModel(wave, flux, ivar, mask)
            desispec.io.write_sky(self.testfile, sky)
            xsky = desispec.io.read_sky(self.testfile)

            self.assertTrue(np.all(sky.wave  == xsky.wave))
            self.assertTrue(np.all(sky.flux  == xsky.flux))
            self.assertTrue(np.all(sky.ivar  == xsky.ivar))
            self.assertTrue(np.all(sky.mask  == xsky.mask))
            self.assertTrue(xsky.flux.dtype.isnative)
            self.assertEqual(sky.mask.dtype, xsky.mask.dtype)

    # fiberflat,fiberflat_ivar,fiberflat_mask,mean_spectrum,wave
    def test_fiberflat_rw(self):
        nspec, nwave, ndiag = 10, 20, 3
        flat = np.random.uniform(size=(nspec, nwave))
        ivar = np.random.uniform(size=(nspec, nwave))
        mask = np.zeros(shape=(nspec, nwave), dtype=int)
        meanspec = np.random.uniform(size=(nwave,))
        wave = np.arange(nwave)

        ff = FiberFlat(wave, flat, ivar, mask, meanspec)

        desispec.io.write_fiberflat(self.testfile, ff)
        xff = desispec.io.read_fiberflat(self.testfile)

        self.assertTrue(np.all(ff.fiberflat == xff.fiberflat))
        self.assertTrue(np.all(ff.ivar == xff.ivar))
        self.assertTrue(np.all(ff.mask == xff.mask))
        self.assertTrue(np.all(ff.meanspec == xff.meanspec))
        self.assertTrue(np.all(ff.wave == xff.wave))

        self.assertTrue(xff.fiberflat.dtype.isnative)
        self.assertTrue(xff.ivar.dtype.isnative)
        self.assertTrue(xff.mask.dtype.isnative)
        self.assertTrue(xff.meanspec.dtype.isnative)
        self.assertTrue(xff.wave.dtype.isnative)

    def test_fibermap_rw(self):
        fibermap = desispec.io.fibermap.empty_fibermap(10)
        for key in fibermap.dtype.names:
            column = fibermap[key]
            fibermap[key] = np.random.random(column.shape).astype(column.dtype)

        desispec.io.write_fibermap(self.testfile, fibermap)

        #- Read without and with header
        fm = desispec.io.read_fibermap(self.testfile)
        self.assertTrue(isinstance(fm, np.ndarray))

        fm, hdr = desispec.io.read_fibermap(self.testfile, header=True)
        self.assertTrue(isinstance(fm, np.ndarray))
        self.assertTrue(isinstance(hdr, fits.Header))

        self.assertEqual(set(fibermap.dtype.names), set(fm.dtype.names))
        for key in fibermap.dtype.names:
            c1 = fibermap[key]
            c2 = fm[key]
            #- Endianness may change, but kind, size, shape, and values are same
            self.assertEqual(c1.dtype.kind, c2.dtype.kind)
            self.assertEqual(c1.dtype.itemsize, c2.dtype.itemsize)
            self.assertEqual(c1.shape, c2.shape)
            self.assertTrue(np.all(c1 == c2))

    def test_native_endian(self):
        for dtype in ('>f8', '<f8', '<f4', '>f4', '>i4', '<i4', '>i8', '<i8'):
            data1 = np.arange(100).astype(dtype)
            data2 = desispec.io.util.native_endian(data1)
            self.assertTrue(data2.dtype.isnative, dtype+' is not native endian')
            self.assertTrue(np.all(data1 == data2))

    def test_findfile(self):
        filenames1 = list()
        filenames2 = list()
        kwargs = {
            'night':'20150510',
            'expid':2,
            'spectrograph':0
        }
        for i in ('sky', 'stdstars'):
            # kwargs['i'] = i
            for j in ('b','r','z'):
                kwargs['band'] = j
                if i == 'sky':
                    kwargs['camera'] = '{band}{spectrograph:d}'.format(**kwargs)
                else:
                    kwargs['camera'] = 'sp{spectrograph:d}'.format(**kwargs)
                filenames1.append(desispec.io.findfile(i,**kwargs))
                filenames2.append(os.path.join(os.environ['DESI_SPECTRO_REDUX'],
                    os.environ['PRODNAME'],'exposures',kwargs['night'],
                    '{expid:08d}'.format(**kwargs),
                    '{i}-{camera}-{expid:08d}.fits'.format(i=i,camera=kwargs['camera'],expid=kwargs['expid'])))
        for k,f in enumerate(filenames1):
            self.assertEqual(os.path.basename(filenames1[k]),
                os.path.basename(filenames2[k]))
            self.assertEqual(filenames1[k],filenames2[k])
            self.assertEqual(desispec.io.filepath2url(filenames1[k]),
                os.path.join('https://portal.nersc.gov/project/desi',
                'collab','spectro','redux',os.environ['PRODNAME'],'exposures',
                kwargs['night'],'{expid:08d}'.format(**kwargs),
                os.path.basename(filenames2[k])))

    @unittest.skipUnless(os.path.exists(os.path.join(os.environ['HOME'],'.netrc')),"No ~/.netrc file detected.")
    def test_download(self):
        #
        # Test by downloading a single file.  This sidesteps any issues
        # with running multiprocessing within the unittest environment.
        #
        filename = desispec.io.findfile('sky',expid=2,night='20150510',camera='b0',spectrograph=0)
        paths = desispec.io.download(filename)
        self.assertEqual(paths[0],filename)
        self.assertTrue(os.path.exists(paths[0]))

#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()
