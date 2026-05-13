#ifndef MELFEATUREPROCESSOR_H
#define MELFEATUREPROCESSOR_H

#include <QVector>

class MelFeatureProcessor
{
public:
    static QVector<double> computeMel(
        const QVector<double> &magnitude,
        int sampleRate,
        int fftSize,
        int melFilterCount,
        int minFreq,
        int maxFreq,
        bool usePower
    );

    static QVector<double> computeLogMel(
        const QVector<double> &magnitude,
        int sampleRate,
        int fftSize,
        int melFilterCount,
        int minFreq,
        int maxFreq,
        double epsilon,
        bool useDb
    );

    static QVector<double> computeMfcc(
        const QVector<double> &magnitude,
        int sampleRate,
        int fftSize,
        int melFilterCount,
        int mfccCount,
        int minFreq,
        int maxFreq,
        bool includeC0
    );

private:
    static double hzToMel(double hz);
    static double melToHz(double mel);

    static QVector<QVector<double>> createMelFilterbank(
        int melFilterCount,
        int fftBinCount,
        int sampleRate,
        int fftSize,
        int minFreq,
        int maxFreq
    );

    static QVector<double> applyMelFilterbank(
        const QVector<double> &spectrum,
        const QVector<QVector<double>> &filterbank
    );

    static QVector<double> dct(
        const QVector<double> &values,
        int coeffCount,
        bool includeC0
    );
};

#endif