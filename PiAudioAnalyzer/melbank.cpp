#include "melbank.h"

#include <algorithm>
#include <cmath>

double MelBank::hzToMel(double hz)
{
    return 2595.0 * std::log10(1.0 + hz / 700.0);
}

double MelBank::melToHz(double mel)
{
    return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
}

QVector<QVector<double>> MelBank::create(int melFilterCount,
                                         int fftBinCount,
                                         int sampleRate,
                                         int fftSize,
                                         int minFreq,
                                         int maxFreq)
{
    QVector<QVector<double>> filterbank;

    if (melFilterCount <= 0 || fftBinCount <= 0 || sampleRate <= 0 || fftSize <= 0) {
        return filterbank;
    }

    int nyquist = sampleRate / 2;

    minFreq = std::max(0, minFreq);
    maxFreq = std::min(maxFreq, nyquist);

    if (maxFreq <= minFreq) {
        maxFreq = nyquist;
    }

    double minMel = hzToMel(minFreq);
    double maxMel = hzToMel(maxFreq);

    QVector<double> melPoints;

    for (int i = 0; i < melFilterCount + 2; ++i) {
        double ratio = static_cast<double>(i) / (melFilterCount + 1);
        melPoints.append(minMel + ratio * (maxMel - minMel));
    }

    QVector<int> binPoints;

    for (double mel : melPoints) {
        double hz = melToHz(mel);
        int bin = static_cast<int>(std::floor((fftSize + 1) * hz / sampleRate));
        bin = std::max(0, std::min(bin, fftBinCount - 1));
        binPoints.append(bin);
    }

    filterbank.resize(melFilterCount);

    for (int m = 1; m <= melFilterCount; ++m) {
        QVector<double> filter(fftBinCount, 0.0);

        int left = binPoints[m - 1];
        int centre = binPoints[m];
        int right = binPoints[m + 1];

        if (centre <= left) {
            centre = left + 1;
        }

        if (right <= centre) {
            right = centre + 1;
        }

        centre = std::min(centre, fftBinCount - 1);
        right = std::min(right, fftBinCount - 1);

        for (int k = left; k < centre; ++k) {
            filter[k] = static_cast<double>(k - left) / std::max(1, centre - left);
        }

        for (int k = centre; k < right; ++k) {
            filter[k] = static_cast<double>(right - k) / std::max(1, right - centre);
        }

        filterbank[m - 1] = filter;
    }

    return filterbank;
}

QVector<double> MelBank::apply(const QVector<double> &spectrum,
                               const QVector<QVector<double>> &filterbank)
{
    QVector<double> melValues;
    melValues.reserve(filterbank.size());

    for (const QVector<double> &filter : filterbank) {
        double sum = 0.0;
        int count = std::min(static_cast<int>(spectrum.size()),
                             static_cast<int>(filter.size()));

        for (int i = 0; i < count; ++i) {
            sum += spectrum[i] * filter[i];
        }

        melValues.append(sum);
    }

    return melValues;
}
