#!/usr/bin/env python

import fitsio
import numpy.ma as ma
from photutils.isophote import EllipseGeometry, Ellipse

img = fitsio.read('data/image.fits')
img = ma.masked_array(img, img==0)

g = EllipseGeometry(x0=266, y0=266, eps=0.240, sma=110, pa=1.0558)
ell = Ellipse(img, geometry=g)
iso = ell.fit_image(3, integrmode='median', sclip=3, nclip=2, linear=False, step=0.1)
