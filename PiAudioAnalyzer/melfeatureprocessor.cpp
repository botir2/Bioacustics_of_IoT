#include "melfeatureprocessor.h"

#include <algorithm>
#include <cmath>

QVector<double> MelFeatureProcessor::computeMel(
    const QVector<double> &magnitude,
    int sampleRate,
    int fftSize,
    int melFilterCount,
    int minFreq,
    int maxFreq,
    bool usePower
)
{
    if (magnitude.isEmpty() || sampleRate <= 0 || fftSize <= 0) {
        return {};
    }

    QVector<double> spectrum;
    spectrum.reserve(magnitude.size());

    for (double v : magnitude) {
        if (usePower) {
            spectrum.append(v * v);
        } else {
            spectrum.append(v);
        }
    }

    QVector<QVector<double>> filterbank =
        createMelFilterbank(
            melFilterCount,
            spectrum.size(),
            sampleRate,
            fftSize,
            minFreq,
            maxFreq
        );

    return applyMelFilterbank(spectrum, filterbank);
}

QVector<double> MelFeatureProcessor::computeLogMel(
    const QVector<double> &magnitude,
    int sampleRate,
    int fftSize,
    int melFilterCount,
    int minFreq,
    int maxFreq,
    double epsilon,
    bool useDb
)
{
    QVector<double> melValues =
        computeMel(
            magnitude,
            sampleRate,
            fftSize,
            melFilterCount,
            minFreq,
            maxFreq,
            true
        );

    QVector<double> logMel;
    logMel.reserve(melValues.size());

    for (double v : melValues) {
        double safeValue = std::max(v, epsilon);

        if (useDb) {
            logMel.append(10.0 * std::log10(safeValue));
        } else {
            logMel.append(std::log(safeValue));
        }
    }

    return logMel;
}

QVector<double> MelFeatureProcessor::computeMfcc(
    const QVector<double> &magnitude,
    int sampleRate,
    int fftSize,
    int melFilterCount,
    int mfccCount,
    int minFreq,
    int maxFreq,
    bool includeC0
)
{
    QVector<double> logMel =
        computeLogMel(
            magnitude,
            sampleRate,
            fftSize,
            melFilterCount,
            minFreq,
            maxFreq,
            1e-6,
            false
        );

    return dct(logMel, mfccCount, includeC0);
}

double MelFeatureProcessor::hzToMel(double hz)
{
    return 2595.0 * std::log10(1.0 + hz / 700.0);
}

double MelFeatureProcessor::melToHz(double mel)
{
    return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
}

QVector<QVector<double>> MelFeatureProcessor::createMelFilterbank(
    int melFilterCount,
    int fftBinCount,
    int sampleRate,
    int fftSize,
    int minFreq,
    int maxFreq
)
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

        if (centre == left) {
            centre = left + 1;
        }

        if (right == centre) {
            right = centre + 1;
        }

        right = std::min(right, fftBinCount - 1);
        centre = std::min(centre, fftBinCount - 1);

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

QVector<double> MelFeatureProcessor::applyMelFilterbank(
    const QVector<double> &spectrum,
    const QVector<QVector<double>> &filterbank
)
{
    QVector<double> melValues;
    melValues.reserve(filterbank.size());

    for (const QVector<double> &filter : filterbank) {
        double sum = 0.0;

        int count = std::min(spectrum.size(), filter.size());

        for (int i = 0; i < count; ++i) {
            sum += spectrum[i] * filter[i];
        }

        melValues.append(sum);
    }

    return melValues;
}

QVector<double> MelFeatureProcessor::dct(
    const QVector<double> &values,
    int coeffCount,
    bool includeC0
)
{
    QVector<double> coeffs;

    if (values.isEmpty() || coeffCount <= 0) {
        return coeffs;
    }

    int n = values.size();
    int startCoeff = includeC0 ? 0 : 1;

    coeffs.reserve(coeffCount);

    const double pi = std::acos(-1.0);

    for (int k = startCoeff; k < startCoeff + coeffCount; ++k) {
        double sum = 0.0;

        for (int i = 0; i < n; ++i) {
            double angle = pi * k * (2.0 * i + 1.0) / (2.0 * n);
            sum += values[i] * std::cos(angle);
        }

        coeffs.append(sum);
    }

    return coeffs;
}