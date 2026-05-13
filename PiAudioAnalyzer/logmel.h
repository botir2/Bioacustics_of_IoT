#ifndef LOGMEL_H
#define LOGMEL_H

#include <QVector>

class LogMel
{
public:
    static QVector<double> compute(const QVector<double> &magnitude,
                                   int sampleRate,
                                   int fftSize,
                                   int melFilterCount,
                                   int minFreq,
                                   int maxFreq,
                                   double epsilon,
                                   bool useDb);
};

#endif
