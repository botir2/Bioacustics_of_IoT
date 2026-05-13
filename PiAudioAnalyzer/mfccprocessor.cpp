#include "mfccprocessor.h"
#include "logmel.h"

#include <algorithm>
#include <cmath>

QVector<double> MfccProcessor::compute(const QVector<double> &magnitude,
                                       int sampleRate,
                                       int fftSize,
                                       int melFilterCount,
                                       int mfccCount,
                                       int minFreq,
                                       int maxFreq,
                                       bool includeC0)
{
    QVector<double> logMelValues =
        LogMel::compute(magnitude,
                        sampleRate,
                        fftSize,
                        melFilterCount,
                        minFreq,
                        maxFreq,
                        1e-6,
                        false);

    return dct(logMelValues, mfccCount, includeC0);
}

QVector<double> MfccProcessor::dct(const QVector<double> &values,
                                   int coefficientCount,
                                   bool includeC0)
{
    QVector<double> coefficients;

    if (values.isEmpty() || coefficientCount <= 0) {
        return coefficients;
    }

    int n = values.size();
    int startCoeff = includeC0 ? 0 : 1;
    int available = std::max(0, n - startCoeff);
    int actualCount = std::min(coefficientCount, available);

    coefficients.reserve(actualCount);

    const double pi = std::acos(-1.0);

    for (int out = 0; out < actualCount; ++out) {
        int k = startCoeff + out;
        double sum = 0.0;

        for (int i = 0; i < n; ++i) {
            double angle = pi * k * (2.0 * i + 1.0) / (2.0 * n);
            sum += values[i] * std::cos(angle);
        }

        coefficients.append(sum);
    }

    return coefficients;
}
