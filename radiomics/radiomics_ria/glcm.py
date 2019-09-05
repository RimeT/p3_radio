# -*- coding: UTF-8 -*-

import numpy
from six.moves import range
from radiomics_ria import base, cMatrices
from scipy import stats


class RadiomicsGLCM(base.RadiomicsFeaturesBase):

	def __init__(self, inputImage, inputMask, **kwargs):
		super(RadiomicsGLCM, self).__init__(inputImage, inputMask, **kwargs)

		self.symmetricalGLCM = kwargs.get('symmetricalGLCM', True)
		self.weightingNorm = kwargs.get('weightingNorm', None)  # manhattan, euclidean, infinity

		self.P_glcm = None
		self.imageArray = self._applyBinning(self.imageArray)
		pass

	def _initCalculation(self, voxelCoordinates=None):
		self.P_glcm = self._calculateMatrix(voxelCoordinates)
		self.P_glcm_2 = self.P_glcm ** 2
		en_prob = self.P_glcm.copy()
		en_prob[en_prob <=0] = 1
		self.P_glcm_en = numpy.where(self.P_glcm > 0, -numpy.log2(en_prob) * self.P_glcm, self.P_glcm)
		self.P_glcm_en[self.P_glcm_en <= 0] = 0.0

		self.coefficients_2 = self.coefficients.copy()
		self.coefficients_en = self.coefficients.copy()
		self._calculateCoefficients()
		self._calculateCoefficients(suffix="_square")
		self._calculateCoefficients(suffix="_entropy")

		# first order
		self.targetVoxelArray = self.P_glcm.astype('float')
		self.targetVoxelArray = self.targetVoxelArray[~numpy.isnan(self.targetVoxelArray)].reshape((1, -1))
		self.discretizedImageArray = self.P_glcm.copy()
		_, p_i = numpy.unique(self.discretizedImageArray, return_counts=True)
		p_i = p_i.reshape((1, -1))
		sumBins = numpy.sum(p_i, 1, keepdims=True).astype('float')
		sumBins[sumBins == 0] = 1  # Prevent division by 0 errors
		p_i = p_i.astype('float') / sumBins
		self.coefficients['p_i'] = p_i

	def _calculateMatrix(self, voxelCoordinates=None):

		self.logger.debug('Calculating GLCM matrix in C')

		Ng = self.coefficients['Ng']

		matrix_args = [
			self.imageArray,
			self.maskArray,
			numpy.array(self.settings.get('distances', [1])),
			Ng,
			self.settings.get('force2D', False),
			self.settings.get('force2Ddimension', 0)
		]
		if self.voxelBased:
			matrix_args += [self.settings.get('kernelRadius', 1), voxelCoordinates]

		P_glcm, angles = cMatrices.calculate_glcm(*matrix_args)

		self.logger.debug('Process calculated matrix')

		# Delete rows and columns that specify gray levels not present in the ROI
		NgVector = range(1, Ng + 1)  # All possible gray values
		GrayLevels = self.coefficients['grayLevels']  # Gray values present in ROI
		emptyGrayLevels = numpy.array(list(set(NgVector) - set(GrayLevels)))  # Gray values NOT present in ROI

		P_glcm = numpy.delete(P_glcm, emptyGrayLevels - 1, 1)
		P_glcm = numpy.delete(P_glcm, emptyGrayLevels - 1, 2)

		# Optionally make GLCMs symmetrical for each angle
		if self.symmetricalGLCM:
			self.logger.debug('Create symmetrical matrix')
			# Transpose and copy GLCM and add it to P_glcm. Numpy.transpose returns a view if possible, use .copy() to ensure
			# a copy of the array is used and not just a view (otherwise erroneous additions can occur)
			P_glcm += numpy.transpose(P_glcm, (0, 2, 1, 3)).copy()

		# Optionally apply a weighting factor
		if self.weightingNorm is not None:
			self.logger.debug('Applying weighting (%s)', self.weightingNorm)
			pixelSpacing = self.inputImage.GetSpacing()[::-1]
			weights = numpy.empty(len(angles))
			for a_idx, a in enumerate(angles):
				if self.weightingNorm == 'infinity':
					weights[a_idx] = numpy.exp(-max(numpy.abs(a) * pixelSpacing) ** 2)
				elif self.weightingNorm == 'euclidean':
					weights[a_idx] = numpy.exp(-numpy.sum((numpy.abs(a) * pixelSpacing) ** 2))  # sqrt ^ 2 = 1
				elif self.weightingNorm == 'manhattan':
					weights[a_idx] = numpy.exp(-numpy.sum(numpy.abs(a) * pixelSpacing) ** 2)
				elif self.weightingNorm == 'no_weighting':
					weights[a_idx] = 1
				else:
					self.logger.warning('weigthing norm "%s" is unknown, W is set to 1', self.weightingNorm)
					weights[a_idx] = 1

			P_glcm = numpy.sum(P_glcm * weights[None, None, None, :], 3, keepdims=True)

		sumP_glcm = numpy.sum(P_glcm, (1, 2))

		# Delete empty angles if no weighting is applied
		if P_glcm.shape[3] > 1:
			emptyAngles = numpy.where(numpy.sum(sumP_glcm, 0) == 0)
			if len(emptyAngles[0]) > 0:  # One or more angles are 'empty'
				self.logger.debug('Deleting %d empty angles:\n%s', len(emptyAngles[0]), angles[emptyAngles])
				P_glcm = numpy.delete(P_glcm, emptyAngles, 3)
				sumP_glcm = numpy.delete(sumP_glcm, emptyAngles, 1)
			else:
				self.logger.debug('No empty angles')

		# Mark empty angles with NaN, allowing them to be ignored in feature calculation
		sumP_glcm[sumP_glcm == 0] = numpy.nan
		# Normalize each glcm
		P_glcm /= sumP_glcm[:, None, None, :]

		return P_glcm

	def _calculateCoefficients(self, suffix=""):

		self.logger.debug('Calculating GLCM coefficients')

		P_glcm = self.P_glcm
		coefficients = self.coefficients
		if "sq" in suffix:
			coefficients = self.coefficients_2
			P_glcm = self.P_glcm_2
		if "en" in suffix:
			coefficients = self.coefficients_en
			P_glcm = self.P_glcm_en

		Ng = coefficients['Ng']
		eps = numpy.spacing(1)

		NgVector = coefficients['grayLevels'].astype('float')
		# shape = (Ng, Ng)
		i, j = numpy.meshgrid(NgVector, NgVector, indexing='ij', sparse=True)

		# shape = (2*Ng-1)
		kValuesSum = numpy.arange(2, (Ng * 2) + 1, dtype='float')
		# shape = (Ng-1)
		kValuesDiff = numpy.arange(0, Ng, dtype='float')

		# marginal row probabilities #shape = (Nv, Ng, 1, angles)
		px = P_glcm.sum(2, keepdims=True)
		# marginal column probabilities #shape = (Nv, 1, Ng, angles)
		py = P_glcm.sum(1, keepdims=True)

		# shape = (Nv, 1, 1, angles)
		mx = i[None, :, :, None] * P_glcm  # multiply x with glcm
		my = j[None, :, :, None] * P_glcm
		ux = numpy.sum(mx, (1, 2), keepdims=True)
		uy = numpy.sum(my, (1, 2), keepdims=True)

		# row and column mean
		rm = numpy.mean(mx, axis=1, keepdims=True) * numpy.ones_like(i[None, :, :, None])
		cm = numpy.mean(my, axis=2, keepdims=True) * numpy.ones_like(j[None, :, :, None])

		# shape = (Nv, 2*Ng-1, angles)
		pxAddy = numpy.array([numpy.sum(P_glcm[:, i + j == k, :], 1) for k in kValuesSum]).transpose((1, 0, 2))
		# shape = (Nv, Ng, angles)
		pxSuby = numpy.array([numpy.sum(P_glcm[:, numpy.abs(i - j) == k, :], 1) for k in kValuesDiff]).transpose(
			(1, 0, 2))

		# shape = (Nv, angles)
		HXY = (-1) * numpy.sum((P_glcm * numpy.log2(P_glcm + eps)), (1, 2))
		coefficients['eps'] = eps
		coefficients['i'] = i
		coefficients['j'] = j
		coefficients['kValuesSum'] = kValuesSum
		coefficients['kValuesDiff'] = kValuesDiff
		coefficients['px'] = px
		coefficients['py'] = py
		coefficients['mx'] = mx
		coefficients['my'] = my
		coefficients['ux'] = ux
		coefficients['uy'] = uy
		coefficients['rm'] = rm
		coefficients['cm'] = cm
		coefficients['pxAddy'] = pxAddy
		coefficients['pxSuby'] = pxSuby
		coefficients['HXY'] = HXY

	def _choose_glcm(self, suffix=""):
		if "sq" in suffix:
			return self.P_glcm_2, self.coefficients_2
		if "en" in suffix:
			return self.P_glcm_en, self.coefficients_en
		return self.P_glcm, self.coefficients

	def getContrastFeatureValue(self, suffix=""):
		P_glcm, coefficients = self._choose_glcm(suffix)
		i = coefficients['i']
		j = coefficients['j']
		w = ((numpy.abs(i - j))[None, :, :, None] ** 2)
		cont = numpy.sum((P_glcm * w), (1, 2))
		return numpy.nanmean(cont, 1)

	def getHomogeneity1FeatureValue(self, nd=False, suffix=""):
		P_glcm, coefficients = self._choose_glcm(suffix)
		pxSuby = coefficients['pxSuby']
		if nd:
			pxSuby[:, 0, :] = 0
		kValuesDiff = coefficients['kValuesDiff']
		invDiff = numpy.sum(pxSuby / (1 + kValuesDiff[None:, None]), 1)
		return numpy.nanmean(invDiff, 1)

	def getHomogeneity2FeatureValue(self, nd=False, suffix=""):
		P_glcm, coefficients = self._choose_glcm(suffix)
		pxSuby = coefficients['pxSuby']
		if nd:
			pxSuby[:, 0, :] = 0
		kValuesDiff = coefficients['kValuesDiff']
		idm = numpy.sum(pxSuby / (1 + (kValuesDiff[None, :, None] ** 2)), 1)
		return numpy.nanmean(idm, 1)

	def getDissimilarityFeatureValue(self, suffix=""):
		P_glcm, coefficients = self._choose_glcm(suffix)
		pxSuby = coefficients['pxSuby']
		kValuesDiff = coefficients['kValuesDiff']
		diffavg = numpy.sum((kValuesDiff[None, :, None] * pxSuby), 1)
		return numpy.nanmean(diffavg, 1)

	def getDmnFeatureValue(self, inverse=False, nd=False, power=2, suffix=""):
		P_glcm, coefficients = self._choose_glcm(suffix)
		pxSuby = coefficients['pxSuby']
		if nd:
			pxSuby[:, 0, :] = 0
		kValuesDiff = coefficients['kValuesDiff']
		Ng = coefficients['Ng']
		if inverse:
			w = 1. / (1. + ((kValuesDiff[None, :, None] ** power) / (Ng ** power)))
		else:
			w = (kValuesDiff[None, :, None] ** power) / (Ng ** power)
		dmn = numpy.sum(pxSuby * w, 1)
		return numpy.nanmean(dmn, 1)

	def getAutocorrelationFeatureValue(self, inverse=False, nd=False, suffix=""):
		P_glcm, coefficients = self._choose_glcm(suffix)
		i = coefficients['i']
		j = coefficients['j']
		if inverse:
			w = 1 / (i * j)
		else:
			w = i * j
		if nd:
			numpy.fill_diagonal(w, 0)
		w = w[None, :, :, None]
		ac = numpy.sum(P_glcm * w, (1, 2))
		return numpy.nanmean(ac, 1)

	def getGaussianFeatureValue(self, inverse=False, nd=False, suffix=""):
		P_glcm, coefficients = self._choose_glcm(suffix)
		gray_levels = coefficients['grayLevels']
		g = len(gray_levels)
		if g == 1:
			return 0
		std = numpy.std(gray_levels, ddof=1) if g != 1 else 1e-2
		avg = numpy.mean(gray_levels)
		Gaussian = lambda x: numpy.exp(-numpy.power(x - avg, 2.) / (2. * numpy.power(std, 2.)))
		w = numpy.fromfunction(lambda i, j: Gaussian(i + 1) * Gaussian(j + 1), (g, g))
		if inverse:
			w = numpy.fromfunction(lambda i, j: 1. / (Gaussian(i + 1) * Gaussian(j + 1)), (g, g))
		if nd:
			numpy.fill_diagonal(w, 0)
		w = w[None, :, :, None]
		if numpy.sum(P_glcm) == 0:
			return 0
		else:
			return numpy.nanmean(numpy.sum(P_glcm * w, (1, 2)), 1)

	def getGaussianLeftPolarFeatureValue(self, inverse=False, nd=False, suffix=""):
		P_glcm, coefficients = self._choose_glcm(suffix)
		gray_levels = coefficients['grayLevels']
		g = len(gray_levels)
		if g == 1:
			return 0
		std = 1e-2 if g ==1 else numpy.std(gray_levels, ddof=1)
		avg = 1.
		Gaussian = lambda x: numpy.exp(-numpy.power(x - avg, 2.) / (2. * numpy.power(std, 2.)))
		w = numpy.fromfunction(lambda i, j: Gaussian(i + 1) * Gaussian(j + 1), (g, g))
		if inverse:
			w = numpy.fromfunction(lambda i, j: 1. / (Gaussian(i + 1) * Gaussian(j + 1)), (g, g))
		if nd:
			numpy.fill_diagonal(w, 0)
		w = w[None, :, :, None]
		if numpy.sum(P_glcm) == 0:
			return 0
		ac = numpy.sum(P_glcm * w, (1, 2))
		return numpy.nanmean(ac, 1)

	def getGaussianRightPolarFeatureValue(self, inverse=False, nd=False, suffix=""):
		P_glcm, coefficients = self._choose_glcm(suffix)
		gray_levels = coefficients['grayLevels']
		g = len(gray_levels)
		if g == 1:
			return 0
		std = 1e-2 if g == 1 else numpy.std(gray_levels, ddof=1)
		avg = g
		Gaussian = lambda x: numpy.exp(-numpy.power(x - avg, 2.) / (2. * numpy.power(std, 2.)))
		w = numpy.fromfunction(lambda i, j: Gaussian(i + 1) * Gaussian(j + 1), (g, g))
		if inverse:
			w = numpy.fromfunction(lambda i, j: 1. / (Gaussian(i + 1) * Gaussian(j + 1)), (g, g))
		if nd:
			numpy.fill_diagonal(w, 0)
		w = w[None, :, :, None]
		if numpy.sum(P_glcm) == 0:
			return 0
		ac = numpy.sum(P_glcm * w, (1, 2))
		return numpy.nanmean(ac, 1)

	def getGaussianLeftFocusFeatureValue(self, inverse=False, nd=False, suffix=""):
		P_glcm, coefficients = self._choose_glcm(suffix)
		gray_levels = coefficients['grayLevels']
		g = len(gray_levels)
		if g == 1:
			return 0
		std = 1e-2 if g == 1 else numpy.std(gray_levels, ddof=1)
		avg = numpy.nanmean(numpy.arange(1, 1 + numpy.ceil(g / 2.)))
		Gaussian = lambda x: numpy.exp(-numpy.power(x - avg, 2.) / (2. * numpy.power(std, 2.)))
		w = numpy.fromfunction(lambda i, j: Gaussian(i + 1) * Gaussian(j + 1), (g, g))
		if inverse:
			w = numpy.fromfunction(lambda i, j: 1. / (Gaussian(i + 1) * Gaussian(j + 1)), (g, g))
		if nd:
			numpy.fill_diagonal(w, 0)
		w = w[None, :, :, None]
		if numpy.sum(P_glcm) == 0:
			return 0
		ac = numpy.sum(P_glcm * w, (1, 2))
		return numpy.nanmean(ac, 1)

	def getGaussianRightFocusFeatureValue(self, inverse=False, nd=False, suffix=""):
		P_glcm, coefficients = self._choose_glcm(suffix)
		gray_levels = coefficients['grayLevels']
		g = len(gray_levels)
		if g == 1:
			return 0
		std = 1e-2 if g == 1 else numpy.std(gray_levels, ddof=1)
		avg = numpy.nanmean(numpy.arange(numpy.floor(g / 2.) + 1, g + 1))
		Gaussian = lambda x: numpy.exp(-numpy.power(x - avg, 2.) / (2. * numpy.power(std, 2.)))
		w = numpy.fromfunction(lambda i, j: Gaussian(i + 1) * Gaussian(j + 1), (g, g))
		if inverse:
			w = numpy.fromfunction(lambda i, j: 1. / (Gaussian(i + 1) * Gaussian(j + 1)), (g, g))
		if nd:
			numpy.fill_diagonal(w, 0)
		w = w[None, :, :, None]
		if numpy.sum(P_glcm) == 0:
			return 0
		ac = numpy.sum(P_glcm * w, (1, 2))
		return numpy.nanmean(ac, 1)

	def getGaussian2FocusFeatureValue(self, inverse=False, nd=False, suffix=""):
		P_glcm, coefficients = self._choose_glcm(suffix)
		gray_levels = coefficients['grayLevels']
		g = len(gray_levels)
		if g == 1:
			return 0
		std = 1e-2 if g == 1 else numpy.std(gray_levels, ddof=1)
		avg1 = numpy.nanmean(numpy.arange(1, 1. + numpy.ceil(g / 2.)))
		avg2 = numpy.nanmean(numpy.arange(numpy.floor(g / 2.) + 1, g + 1))
		Gaussian = lambda x, mu: numpy.exp(-numpy.power(x - mu, 2.) / (2. * numpy.power(std, 2.)))
		w = numpy.fromfunction(
			lambda i, j: Gaussian(i + 1, avg1) * Gaussian(j + 1, avg1) + Gaussian(i + 1, avg2) * Gaussian(j + 1, avg2), (g, g))
		if inverse:
			w = numpy.fromfunction(
				lambda i, j: 1. / (Gaussian(i + 1, avg1) * Gaussian(j + 1, avg1)) + 1. / (Gaussian(i + 1, avg2) * Gaussian(j + 1, avg2)), (g, g))
		if nd:
			numpy.fill_diagonal(w, 0)
		w = w[None, :, :, None]
		if numpy.sum(P_glcm) == 0:
			return 0
		ac = numpy.sum(P_glcm * w, (1, 2))
		return numpy.nanmean(ac, 1)

	def getGaussian2PolarFeatureValue(self, inverse=False, nd=False, suffix=""):
		P_glcm, coefficients = self._choose_glcm(suffix)
		gray_levels = coefficients['grayLevels']
		g = len(gray_levels)
		if g == 1:
			return 0
		std = numpy.std(gray_levels, ddof=1) if g !=1 else 1e-2
		avg1 = 1
		avg2 = g
		Gaussian = lambda x, mu: numpy.exp(-numpy.power(x - mu, 2.) / (2. * numpy.power(std, 2.)))
		w = numpy.fromfunction(
			lambda i, j: Gaussian(i + 1, avg1) * Gaussian(j + 1, avg1) + Gaussian(i + 1, avg2) * Gaussian(j + 1, avg2), (g, g))
		if inverse:
			w = numpy.fromfunction(
				lambda i, j: 1. / (Gaussian(i + 1, avg1) * Gaussian(j + 1, avg1)) + 1. / (Gaussian(i + 1, avg2) * Gaussian(j + 1, avg2)), (g, g))
		if nd:
			numpy.fill_diagonal(w, 0)
		if numpy.sum(P_glcm) == 0:
			return 0
		w = w[None, :, :, None]
		ac = numpy.sum(P_glcm * w, (1, 2))
		return numpy.nanmean(ac, 1)

	def getClusterFeatureValue(self, nd=False, inverse=False, power=3, suffix=""):
		P_glcm, coefficients = self._choose_glcm(suffix)
		i = coefficients['i']
		j = coefficients['j']
		rm = coefficients['rm']
		cm = coefficients['cm']
		w = (numpy.abs((i + j)[None, :, :, None] - rm - cm)) ** power
		if inverse:
			w[w==0.0] = 1e-8
			w = 1. / w
		if nd:
			for x in range(w.shape[-1]):
				numpy.fill_diagonal(w[0, :, :, x], 0)
		cp = numpy.sum((P_glcm * w), (1, 2))
		return numpy.nanmean(cp, 1)

	def getAverageFeatureValue(self, suffix=""):
		P_glcm, coefficients = self._choose_glcm(suffix)
		i = coefficients['i']
		j = coefficients['j']
		w = i + numpy.zeros_like(j)
		w = w[None, :, :, None]
		ac = numpy.sum(P_glcm * w, (1, 2))
		return numpy.nanmean(ac, 1)

	def getVarianceFeatureValue(self, suffix=""):
		P_glcm, coefficients = self._choose_glcm(suffix)
		i = coefficients['i']
		j = coefficients['j']
		sum_arr = numpy.nansum(i[None, :, :, None] * P_glcm, (1, 2))
		w = (i[None, :, :, None] + numpy.zeros_like(j[None, :, :, None]) - sum_arr) ** 2
		ac = numpy.sum(P_glcm * w, (1, 2))
		return numpy.nanmean(ac, 1)

	def getCorrelationFeatureValue(self, suffix=""):
		P_glcm, coefficients = self._choose_glcm(suffix)
		i = coefficients['i']
		j = coefficients['j']
		mu = numpy.nansum(i[None, :, :, None] * P_glcm, (1, 2))
		sig = numpy.nansum(P_glcm * (i[None, :, :, None] - mu) ** 2)
		if sig == 0:
			sig = 1e-8
		w = ((i[None, :, :, None] + numpy.zeros_like(j[None, :, :, None]) - mu) * (j[None, :, :, None] + numpy.zeros_like(i[None, :, :, None]) - mu)) / sig
		ac = numpy.sum(P_glcm * w, (1, 2))
		return numpy.nanmean(ac, 1)

	# SUM

	def getSumAverageFeatureValue(self, inverse=False):
		P_glcm, coefficients = self._choose_glcm(suffix="")
		pxAddy = coefficients['pxAddy']
		kValuesSum = coefficients['kValuesSum']
		if inverse:
			kValuesSum = 1. / kValuesSum
		sumavg = numpy.sum((kValuesSum[None, :, None] * pxAddy), 1)
		return numpy.nanmean(sumavg, 1)

	def getSumEnergyFeatureValue(self, inverse=False):
		P_glcm, coefficients = self._choose_glcm(suffix="")
		pxAddy = coefficients['pxAddy'] ** 2
		kValuesSum = coefficients['kValuesSum']
		if inverse:
			kValuesSum = 1. / kValuesSum
		sumavg = numpy.sum((kValuesSum[None, :, None] * pxAddy), 1)
		return numpy.nanmean(sumavg, 1)

	def getSumEntropyFeatureValue(self):
		P_glcm, coefficients = self._choose_glcm(suffix="")
		pxAddy = coefficients['pxAddy']
		eps = coefficients['eps']
		sumentr = (-1) * numpy.sum((pxAddy * numpy.log2(pxAddy + eps)), 1)
		return numpy.nanmean(sumentr, 1)

	def getSumVarianceFeatureValue(self, inverse=False):
		P_glcm, coefficients = self._choose_glcm(suffix="")
		kValuesSum = coefficients['kValuesSum']
		pxAddy = coefficients['pxAddy']
		et = self.getSumEntropyFeatureValue()
		we = (kValuesSum - et) ** 2
		if inverse:
			we = 1. / we
		sumvar = numpy.sum(we[None, :, None] * pxAddy, 1)
		return numpy.nanmean(sumvar, 1)

	# DIFFERENCE

	def getDifferenceAverageFeatureValue(self, inverse=False):
		P_glcm, coefficients = self._choose_glcm(suffix="")
		pxAddy = coefficients['pxSuby']
		kValuesDif = coefficients['kValuesDiff']
		if inverse:
			kValuesDif = numpy.divide(numpy.ones_like(kValuesDif),
									  kValuesDif,
									  out=numpy.zeros_like(kValuesDif),
									  where=kValuesDif!=0)
		sumavg = numpy.sum((kValuesDif[None, :, None] * pxAddy), 1)
		return numpy.nanmean(sumavg, 1)

	def getDifferenceEnergyFeatureValue(self, inverse=False):
		P_glcm, coefficients = self._choose_glcm(suffix="")
		pxAddy = coefficients['pxSuby'] ** 2
		kValuesDif = coefficients['kValuesDiff']
		if inverse:
			kValuesDif = numpy.divide(numpy.ones_like(kValuesDif),
									  kValuesDif,
									  out=numpy.zeros_like(kValuesDif),
									  where=kValuesDif!=0)
		sumavg = numpy.sum((kValuesDif[None, :, None] * pxAddy), 1)
		return numpy.nanmean(sumavg, 1)

	def getDifferenceEntropyFeatureValue(self):
		P_glcm, coefficients = self._choose_glcm(suffix="")
		pxSuby = coefficients['pxSuby']
		eps = coefficients['eps']
		difent = (-1) * numpy.sum((pxSuby * numpy.log2(pxSuby + eps)), 1)
		return numpy.nanmean(difent, 1)

	def getDifferenceVarianceFeatureValue(self, inverse=False):
		P_glcm, coefficients = self._choose_glcm(suffix="")
		pxSuby = coefficients['pxSuby']
		kValuesDiff = coefficients['kValuesDiff']
		et = self.getDifferenceEntropyFeatureValue()
		we = (kValuesDiff - et) ** 2
		if inverse:
			we = 1. / we
		diffvar = numpy.sum(we[None, :, None] * pxSuby, 1)
		return numpy.nanmean(diffvar, 1)

	# ICM

	def getImc1FeatureValue(self):
		P_glcm, coefficients = self._choose_glcm(suffix="")
		HXY = coefficients['HXY']
		eps = coefficients['eps']
		px = coefficients['px']
		py = coefficients['py']
		# entropy of px # shape = (Nv, angles)
		HX = (-1) * numpy.sum((px * numpy.log2(px + eps)), (1, 2))
		# entropy of py # shape = (Nv, angles)
		HY = (-1) * numpy.sum((py * numpy.log2(py + eps)), (1, 2))
		# shape = (Nv, angles)
		HXY1 = (-1) * numpy.sum((P_glcm * numpy.log2(px * py + eps)), (1, 2))
		div = numpy.fmax(HX, HY)
		imc1 = HXY - HXY1
		imc1[div != 0] /= div[div != 0]
		imc1[div == 0] = 0  # Set elements that would be divided by 0 to 0
		return numpy.nanmean(imc1, 1)

	def getImc2FeatureValue(self):
		P_glcm, coefficients = self._choose_glcm(suffix="")
		HXY = coefficients['HXY']
		eps = coefficients['eps']
		px = coefficients['px']
		py = coefficients['py']
		# shape = (Nv, angles)
		HXY2 = (-1) * numpy.sum(((px * py) * numpy.log2(px * py + eps)), (1, 2))
		imc2 = (1 - numpy.e ** (-2 * (HXY2 - HXY))) ** 0.5
		imc2[HXY2 == HXY] = 0
		return numpy.nanmean(imc2, 1)

	# first order

	@staticmethod
	def _moment(a, moment=1):
		if moment == 1:
			return numpy.float(0.0)
		else:
			mn = numpy.nanmean(a, 1, keepdims=True)
			s = numpy.power((a - mn), moment)
			return numpy.nanmean(s, 1)

	def getMeanFeatureValue(self):
		return numpy.nanmean(self.targetVoxelArray, 1)

	def getMedianFeatureValue(self):
		return numpy.nanmedian(self.targetVoxelArray, 1)

	def getModeFeatureValue(self):
		out = stats.mode(self.targetVoxelArray, 1)[0][0]
		return out

	def getGeoMean1FeatureValue(self):
		arr = abs(self.targetVoxelArray.copy())
		arr[arr == 0 ] = 1.
		out = numpy.e ** numpy.nanmean(numpy.log2(arr))
		return out

	def getGeoMean2FeatureValue(self):
		arr = self.targetVoxelArray.copy()
		arr_abs = abs(arr)
		arr_abs[arr_abs == 0 ] = 1.
		arr_log = numpy.log2(arr_abs)
		arr_log = numpy.where(arr_log < 0, -1. * arr_log, arr_log)
		out = numpy.e ** numpy.nanmean(arr_log)
		return out

	def getGeoMean3FeatureValue(self):
		arr = self.targetVoxelArray.copy()
		mv = numpy.min(self.targetVoxelArray)
		dev = numpy.abs(mv)
		if mv <= 0:
			arr += dev + 1
		out = numpy.e ** numpy.nanmean(numpy.log(arr))
		return out

	def getHarmonicMeanFeatureValue(self):
		arr = self.targetVoxelArray.copy()
		arr[arr == 0] = 1.
		size = arr.shape[-1]
		out = size / numpy.sum(1.0 / arr, axis=-1)
		return out

	def getTrimmedMeanFeatureValue(self, percent=0.05):
		arr = self.targetVoxelArray.copy()
		out = stats.trim_mean(arr, percent, 1)
		return out

	def getTrimMeanFeatureValue(self):
		arr = self.targetVoxelArray.copy()
		out = (numpy.nanpercentile(arr, 25, axis=1) + numpy.nanpercentile(arr, 75, axis=1) +
			   numpy.nanpercentile(arr, 50, axis=1) * 2 ) / 4
		return out

	def getMeanAbsoluteDeviationMeanFeatureValue(self):
		u_x = numpy.nanmean(self.targetVoxelArray, 1, keepdims=True)
		return numpy.nanmean(numpy.absolute(self.targetVoxelArray - u_x), 1)

	def getMeanAbsoluteDeviationMedianFeatureValue(self):
		d_x = numpy.nanmedian(self.targetVoxelArray, 1, keepdims=True)
		return numpy.nanmean(numpy.absolute(self.targetVoxelArray - d_x), 1)

	def getMedianAbsoluteDeviationFromMeanFeatureValue(self):
		u_x = numpy.nanmean(self.targetVoxelArray, 1, keepdims=True)
		return numpy.nanmedian(numpy.absolute(self.targetVoxelArray - u_x), 1)

	def getMedianAbsoluteDeviationFromMedianFeatureValue(self):
		d_x = numpy.nanmedian(self.targetVoxelArray, 1, keepdims=True)
		return numpy.nanmedian(numpy.absolute(self.targetVoxelArray - d_x), 1)

	def getMaxAbsoluteDeviationFromMeanFeatureValue(self):
		u_x = numpy.nanmean(self.targetVoxelArray, 1, keepdims=True)
		return numpy.nanmax(numpy.absolute(self.targetVoxelArray - u_x), 1)

	def getMaxAbsoluteDeviationFromMedianFeatureValue(self):
		d_x = numpy.nanmedian(self.targetVoxelArray, 1, keepdims=True)
		return numpy.nanmax(numpy.absolute(self.targetVoxelArray - d_x), 1)

	def getMedianAbsoluteDeviationFeatureValue(self):
		d_x = numpy.nanmedian(self.targetVoxelArray, 1, keepdims=True)
		return numpy.nanmedian(numpy.absolute(self.targetVoxelArray - d_x), 1) * 1.4826

	def getRootMeanSquaredFeatureValue(self):
		if self.targetVoxelArray.size == 0:
			return 0
		Nvox = numpy.sum(~numpy.isnan(self.targetVoxelArray), 1).astype('float')
		return numpy.sqrt(numpy.nansum(self.targetVoxelArray ** 2, 1) / Nvox)

	def getMinimumFeatureValue(self):
		return numpy.nanmin(self.targetVoxelArray, 1)

	def getMaximumFeatureValue(self):
		return numpy.nanmax(self.targetVoxelArray, 1)

	def get25PercentileFeatureValue(self):
		return numpy.nanpercentile(self.targetVoxelArray, 25, axis=1)

	def get75PercentileFeatureValue(self):
		return numpy.nanpercentile(self.targetVoxelArray, 75, axis=1)

	def getInterquartileRangeFeatureValue(self):
		return numpy.absolute(numpy.nanpercentile(self.targetVoxelArray, 75, 1) -
							  numpy.nanpercentile(self.targetVoxelArray, 25, 1))

	def getLowerNotchFeatureValue(self):
		return self.get25PercentileFeatureValue() - self.getInterquartileRangeFeatureValue() * 1.5

	def getUpperNotchFeatureValue(self):
		return self.get75PercentileFeatureValue() + self.getInterquartileRangeFeatureValue() * 1.5

	def getRangeFeatureValue(self):
		return numpy.absolute(numpy.nanmax(self.targetVoxelArray, 1) - numpy.nanmin(self.targetVoxelArray, 1))

	def getDecilesFeatureValue(self, percent):
		return numpy.nanpercentile(self.targetVoxelArray, percent, axis=1)

	def getStandardDeviationFeatureValue(self):
		return numpy.nanstd(self.targetVoxelArray, axis=1)

	def getSkewnessFeatureValue(self):
		m2 = self._moment(self.targetVoxelArray, 2)
		m3 = self._moment(self.targetVoxelArray, 3)
		m2[m2 == 0] = 1  # Flat Region, prevent division by 0 errors
		m3[m2 == 0] = 0  # ensure Flat Regions are returned as 0
		return m3 / m2 ** 1.5

	def getKurtosisFeatureValue(self):
		m2 = self._moment(self.targetVoxelArray, 2)
		m4 = self._moment(self.targetVoxelArray, 4)
		m2[m2 == 0] = 1  # Flat Region, prevent division by 0 errors
		m4[m2 == 0] = 0  # ensure Flat Regions are returned as 0
		return m4 / m2 ** 2.0

	def getEnergyFeatureValue(self):
		return numpy.nansum(self.targetVoxelArray ** 2, 1)

	def getUniformityFeatureValue(self):
		p_i = self.coefficients['p_i']
		return numpy.nansum(p_i ** 2, 1)

	def getEntropyFeatureValue(self):
		p_i = self.coefficients['p_i']
		eps = numpy.spacing(1)
		return -1.0 * numpy.sum(p_i * numpy.log2(p_i + eps), 1)
