import numpy
import SimpleITK as sitk

from radiomics_ria import base, cShape, deprecated


class RadiomicsShape(base.RadiomicsFeaturesBase):

	def __init__(self, inputImage, inputMask, **kwargs):
		assert inputMask.GetDimension() == 3, 'Shape features are only available in 3D. If 2D, use shape2D instead'
		super(RadiomicsShape, self).__init__(inputImage, inputMask, **kwargs)

	def _initSegmentBasedCalculation(self):
		self.pixelSpacing = numpy.array(self.inputImage.GetSpacing()[::-1])

		self.maskArray = (sitk.GetArrayFromImage(self.inputMask) == self.label)
		self.imageArray = self._applyBinning(self.imageArray)

		pass
	
	def update_focus(self, focus):

		mask = (self.imageArray == focus) & (self.maskArray)
		mask = mask * 1

		cpif = sitk.ConstantPadImageFilter()
		padding = numpy.tile(1, 3)
		try:
			cpif.SetPadLowerBound(padding)
			cpif.SetPadUpperBound(padding)
		except TypeError:
			# newer versions of SITK/python want a tuple or list
			cpif.SetPadLowerBound(padding.tolist())
			cpif.SetPadUpperBound(padding.tolist())
		mask_itk = sitk.GetImageFromArray(mask)
		mask_itk.SetSpacing(self.pixelSpacing)
		mask = sitk.GetArrayFromImage(cpif.Execute(mask_itk))
		
		self.labelledVoxelCoordinates = numpy.where(mask != 0)
		Np = len(self.labelledVoxelCoordinates[0])

		self.SurfaceArea, self.Volume, self.diameters = cShape.calculate_coefficients(mask, self.pixelSpacing)
		# self.Volume = self.pixelSpacing[0] * self.pixelSpacing[1] * self.pixelSpacing[2] * Np

		# if self.Volume == 0:
		# 	self.Volume = 0.001

		# if self.SurfaceArea == 0:
		# 	self.SurfaceArea = 0.001

		# Compute eigenvalues and -vectors
		if self.Volume != 0:
			coordinates = numpy.array(self.labelledVoxelCoordinates, dtype='int').transpose((1, 0))
			physicalCoordinates = coordinates * self.pixelSpacing[None, :]
			try:
				physicalCoordinates -= numpy.mean(physicalCoordinates, axis=0)  # Centered at 0
			except Exception:
				pass
			physicalCoordinates /= numpy.sqrt(Np)
			covariance = numpy.dot(physicalCoordinates.T.copy(), physicalCoordinates)
			self.eigenValues = numpy.linalg.eigvals(covariance)

			# Correct machine precision errors causing very small negative eigen values in case of some 2D segmentations
			machine_errors = numpy.bitwise_and(self.eigenValues < 0, self.eigenValues > -1e-10)
			if numpy.sum(machine_errors) > 0:
				# self.logger.warning('Encountered %d eigenvalues < 0 and > -1e-10, rounding to 0', numpy.sum(machine_errors))
				self.eigenValues[machine_errors] = 0

			self.eigenValues.sort()  # Sort the eigenValues from small to large

	def getVolumeFeatureValue(self):
		return self.Volume

	def getSurfaceAreaFeatureValue(self):
		return self.SurfaceArea

	def getSurfaceVolumeRatioFeatureValue(self):
		if self.Volume == 0:
			return 0
		return self.SurfaceArea / self.Volume

	def getSphericityFeatureValue(self):
		if self.Volume == 0:
			return 0
		return (36 * numpy.pi * self.Volume ** 2) ** (1.0 / 3.0) / self.SurfaceArea

	def getCompactness1FeatureValue(self):
		if self.Volume == 0:
			return 0
		return self.Volume / (self.SurfaceArea ** (3.0 / 2.0) * numpy.sqrt(numpy.pi))

	def getCompactness2FeatureValue(self):
		if self.Volume == 0:
			return 0
		return (36.0 * numpy.pi) * (self.Volume ** 2.0) / (self.SurfaceArea ** 3.0)

	def getSphericalDisproportionFeatureValue(self):
		if self.Volume == 0:
			return 0
		return self.SurfaceArea / (36 * numpy.pi * self.Volume ** 2) ** (1.0 / 3.0)