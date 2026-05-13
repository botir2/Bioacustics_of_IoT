#include "logmel.h"
#include "melspec.h"

#include <algorithm>
#include <cmath>

QVector<double> LogMel::compute(const QVector<double> &magnitude,
                                int sampleRate,
                                int fftSize,
                                int melFilterCount,
                                int minFreq,
                                int maxFreq,
                                double epsilon,
                                bool useDb)
{
    QVector<double> melValues =
        MelSpec::compute(magnitude,
                         sampleRate,
                         fftSize,
                         melFilterCount,
                         minFreq,
                         maxFreq,
                         true);

    QVector<double> logMelValues;
    logMelValues.reserve(melValues.size());

    epsilon = std::max(epsilon, 1e-12);

    for (double v : melValues) {
        double safeValue = std::max(v, epsilon);

        if (useDb) {
            logMelValues.append(10.0 * std::log10(safeValue));
        } else {
            logMelValues.append(std::log(safeValue));
        }
    }

    return logMelValues;
}
