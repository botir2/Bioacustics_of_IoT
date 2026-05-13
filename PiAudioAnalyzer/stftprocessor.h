#ifndef STFTPROCESSOR_H
#define STFTPROCESSOR_H

#include <QVector>

struct StftFrame
{
    QVector<double> waveform;
    QVector<double> magnitude;

    double rms = 0.0;
    double peak = 0.0;
    double dominantFrequency = 0.0;
};

class StftProcessor
{
public:
    void configure(int sampleRate, int windowSize, int hopSize);
    void reset();

    QVector<StftFrame> processSamples(const QVector<double> &newSamples);

private:
    int m_sampleRate = 44100;
    int m_windowSize = 1024;
    int m_hopSize = 512;

    QVector<double> m_buffer;

    StftFrame processOneFrame(const QVector<double> &window);
};

#endif
