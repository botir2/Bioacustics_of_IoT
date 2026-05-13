#ifndef MFCCPROCESSOR_H
#define MFCCPROCESSOR_H

#include <QVector>

class MfccProcessor
{
public:
    static QVector<double> compute(const QVector<double> &magnitude,
                                   int sampleRate,
                                   int fftSize,
                                   int melFilterCount,
                                   int mfccCount,
                                   int minFreq,
                                   int maxFreq,
                                   bool includeC0);

private:
    static QVector<double> dct(const QVector<double> &values,
                               int coefficientCount,
                               bool includeC0);
};

#endif
