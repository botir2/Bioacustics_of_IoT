#ifndef NOISEREDUCER_H
#define NOISEREDUCER_H

#include <QVector>

class NoiseReducer
{
public:
    NoiseReducer();

    QVector<float> reduce(const QVector<float>& inputSamples,
                          float threshold,
                          float strength,
                          float smoothing,
                          bool nonStationary);
};

#endif // NOISEREDUCER_H
