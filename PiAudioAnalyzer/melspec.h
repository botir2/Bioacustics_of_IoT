#ifndef MELSPEC_H
#define MELSPEC_H

#include <QVector>

class MelSpec
{
public:
    static QVector<double> compute(const QVector<double> &magnitude,
                                   int sampleRate,
                                   int fftSize,
                                   int melFilterCount,
                                   int minFreq,
                                   int maxFreq,
                                   bool usePower);
};

#endif
