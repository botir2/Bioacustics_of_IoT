#ifndef MELBANK_H
#define MELBANK_H

#include <QVector>

class MelBank
{
public:
    static QVector<QVector<double>> create(int melFilterCount,
                                           int fftBinCount,
                                           int sampleRate,
                                           int fftSize,
                                           int minFreq,
                                           int maxFreq);

    static QVector<double> apply(const QVector<double> &spectrum,
                                 const QVector<QVector<double>> &filterbank);

    static double hzToMel(double hz);
    static double melToHz(double mel);
};

#endif
