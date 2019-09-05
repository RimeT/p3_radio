import numpy
from six.moves import range
from scipy import stats
from radiomics_ria import base, cMatrices, deprecated


class RadiomicsFirstOrder(base.RadiomicsFeaturesBase):

    def __init__(self, inputImage, inputMask, **kwargs):
        super(RadiomicsFirstOrder, self).__init__(inputImage, inputMask, **kwargs)
        self.pixelSpacing = inputImage.GetSpacing()
        self.voxelArrayShift = kwargs.get('voxelArrayShift', 0)
        # self.discretizedImageArray = self._applyBinning(self.imageArray.copy())
        self.discretizedImageArray = self.imageArray.copy()

    def _initCalculation(self, voxelCoordinates=None):
        self.targetVoxelArray = self.imageArray[self.maskArray].astype('float')
        self.targetVoxelArray = self.targetVoxelArray[~numpy.isnan(self.targetVoxelArray)].reshape((1, -1))
        _, p_i = numpy.unique(self.discretizedImageArray[self.maskArray], return_counts=True)
        p_i = p_i.reshape((1, -1))
        sumBins = numpy.sum(p_i, 1, keepdims=True).astype('float')
        sumBins[sumBins == 0] = 1  # Prevent division by 0 errors
        p_i = p_i.astype('float') / sumBins
        self.coefficients['p_i'] = p_i

    @staticmethod
    def _moment(a, moment=1):
        r"""
        Calculate n-order moment of an array for a given axis
        """

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

    def getRobustMeanAbsoluteDeviationFeatureValue(self):
        r"""
        **13. Robust Mean Absolute Deviation (rMAD)**

        .. math::
          \textit{rMAD} = \frac{1}{N_{10-90}}\displaystyle\sum^{N_{10-90}}_{i=1}
          {|\textbf{X}_{10-90}(i)-\bar{X}_{10-90}|}

        Robust Mean Absolute Deviation is the mean distance of all intensity values
        from the Mean Value calculated on the subset of image array with gray levels in between, or equal
        to the 10\ :sup:`th` and 90\ :sup:`th` percentile.
        """

        prcnt10 = numpy.percentile(self.targetVoxelArray, 10)
        prcnt90 = numpy.percentile(self.targetVoxelArray, 90)
        percentileArray = self.targetVoxelArray[(self.targetVoxelArray >= prcnt10) * (self.targetVoxelArray <= prcnt90)]

        v2 = percentileArray - numpy.mean(percentileArray)
        v1 = numpy.absolute(v2)
        value = numpy.mean(v1)
        return value

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
        return numpy.sqrt(numpy.nansum(self.targetVoxelArray ** 2, 1) / len(self.targetVoxelArray))

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

    def getVarianceFeatureValue(self):
        return numpy.nanstd(self.targetVoxelArray, 1) ** 2

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