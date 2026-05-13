#ifndef STFTNOISEREDUCER_H
#define STFTNOISEREDUCER_H

#include <QVector>

class StftNoiseReducer
{
public:
    StftNoiseReducer();

    void reset();

    QVector<double> processMagnitude(const QVector<double> &magnitude,
                                      double threshold,
                                      double strength,
                                      double smoothing,
                                      bool nonStationary);

private:
    QVector<double> noiseFloor;
    QVector<double> previousOutput;

    void ensureSize(int size);

    double estimateFrameNoise(const QVector<double> &magnitude) const;
};

#endif