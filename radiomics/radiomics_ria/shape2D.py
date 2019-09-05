import numpy
import SimpleITK as sitk

from radiomics_ria import base, cShape, deprecated


class RadiomicsShape2D(base.RadiomicsFeaturesBase):

  def __init__(self, inputImage, inputMask, **kwargs):
    super(RadiomicsShape2D, self).__init__(inputImage, inputMask, **kwargs)

  def _initSegmentBasedCalculation(self):
    self.pixelSpacing = numpy.array(self.inputImage.GetSpacing()[::-1])
    self.maskArray = (sitk.GetArrayFromImage(self.inputMask) == self.label)
    self.imageArray = self._applyBinning(self.imageArray)

  def update_focus(self, focus):
    Nd = self.inputMask.GetDimension()
    if Nd == 3:
      if not self.settings.get('force2D', False):
        raise ValueError('Shape2D is can only be calculated when input is 2D or 3D with `force2D=True`')

      force2DDimension = self.settings.get('force2Ddimension', 0)
      axes = [0, 1, 2]
      axes.remove(force2DDimension)

      self.pixelSpacing = numpy.array(self.inputImage.GetSpacing()[::-1])[(axes,)]

      if self.maskArray.shape[force2DDimension] > 1:
        raise ValueError('Size of the mask in dimension %i is more than 1, cannot compute 2D shape')

      # Drop the 2D axis, ensuring the input is truly 2D
      self.maskArray = numpy.squeeze(self.maskArray, axis=force2DDimension)
    elif Nd == 2:
      self.pixelSpacing = numpy.array(self.inputImage.GetSpacing()[::-1])
    else:
      raise ValueError('Shape2D is can only be calculated when input is 2D or 3D with `force2D=True`')

    mask = (self.imageArray != focus) & (self.maskArray)
    mask = mask * 1
    maskITK = sitk.GetImageFromArray(mask)

    maskArray = numpy.pad(maskITK, pad_width=1, mode='constant', constant_values=0)

    self.labelledPixelCoordinates = numpy.where(maskArray != 0)

    # Volume, Surface Area and eigenvalues are pre-calculated

    # Compute Surface Area and volume
    self.Perimeter, self.Surface, self.Diameter = cShape.calculate_coefficients2D(maskArray, self.pixelSpacing)

    # Compute eigenvalues and -vectors
    Np = len(self.labelledPixelCoordinates[0])
    coordinates = numpy.array(self.labelledPixelCoordinates, dtype='int').transpose((1, 0))  # Transpose equals zip(*a)
    physicalCoordinates = coordinates * self.pixelSpacing[None, :]
    physicalCoordinates -= numpy.mean(physicalCoordinates, axis=0)  # Centered at 0
    physicalCoordinates /= numpy.sqrt(Np)
    covariance = numpy.dot(physicalCoordinates.T.copy(), physicalCoordinates)
    self.eigenValues = numpy.linalg.eigvals(covariance)

    # Correct machine precision errors causing very small negative eigen values in case of some 2D segmentations
    machine_errors = numpy.bitwise_and(self.eigenValues < 0, self.eigenValues > -1e-10)
    if numpy.sum(machine_errors) > 0:
      self.logger.warning('Encountered %d eigenvalues < 0 and > -1e-10, rounding to 0', numpy.sum(machine_errors))
      self.eigenValues[machine_errors] = 0

    self.eigenValues.sort()  # Sort the eigenValues from small to large

    self.logger.debug('Shape feature class initialized')

  def getMeshSurfaceFeatureValue(self):
    return self.Surface

  def getPixelSurfaceFeatureValue(self):
    y, x = self.pixelSpacing
    Np = len(self.labelledPixelCoordinates[0])
    return Np * (x * y)

  def getPerimeterFeatureValue(self):
    return self.Perimeter

  def getPerimeterSurfaceRatioFeatureValue(self):
    return self.Perimeter / self.Surface

  def getSphericityFeatureValue(self):
    return (2 * numpy.sqrt(numpy.pi * self.Surface)) / self.Perimeter

  def getSphericalDisproportionFeatureValue(self):
    return 1.0 / self.getSphericityFeatureValue()

  # def getCompactness1FeatureValue(self):
  #   return self.Volume / (self.SurfaceArea ** (3.0 / 2.0) * numpy.sqrt(numpy.pi))
  #
  # def getCompactness2FeatureValue(self):
  #   return (36.0 * numpy.pi) * (self.Volume ** 2.0) / (self.SurfaceArea ** 3.0)