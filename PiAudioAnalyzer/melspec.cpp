#include "melspec.h"
#include "melbank.h"

QVector<double> MelSpec::compute(const QVector<double> &magnitude,
                                 int sampleRate,
                                 int fftSize,
                                 int melFilterCount,
                                 int minFreq,
                                 int maxFreq,
                                 bool usePower)
{
    if (magnitude.isEmpty()) {
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
        MelBank::create(melFilterCount,
                        spectrum.size(),
                        sampleRate,
                        fftSize,
                        minFreq,
                        maxFreq);

    return MelBank::apply(spectrum, filterbank);
}
