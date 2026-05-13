#include "stftprocessor.h"
#include "ffthelper.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

void StftProcessor::configure(int sampleRate, int windowSize, int hopSize)
{
    m_sampleRate = sampleRate;
    m_windowSize = windowSize;
    m_hopSize = hopSize;
    reset();
}

void StftProcessor::reset()
{
    m_buffer.clear();
}

QVector<StftFrame> StftProcessor::processSamples(const QVector<double> &newSamples)
{
    QVector<StftFrame> frames;

    for (double sample : newSamples) {
        m_buffer.append(sample);
    }

    while (m_buffer.size() >= m_windowSize) {
        QVector<double> window;
        window.reserve(m_windowSize);

        for (int i = 0; i < m_windowSize; ++i) {
            window.append(m_buffer[i]);
        }

        frames.append(processOneFrame(window));

        m_buffer.remove(0, m_hopSize);
    }

    return frames;
}

StftFrame StftProcessor::processOneFrame(const QVector<double> &window)
{
    StftFrame frame;
    frame.waveform = window;

    std::vector<std::complex<double>> buffer(m_windowSize);

    double peak = 0.0;
    double energy = 0.0;

    const double pi = std::acos(-1.0);

    for (int i = 0; i < m_windowSize; ++i) {
        double sample = window[i];

        peak = std::max(peak, std::abs(sample));
        energy += sample * sample;

        double hann = 0.5 * (1.0 - std::cos(2.0 * pi * i / (m_windowSize - 1)));
        buffer[i] = std::complex<double>(sample * hann, 0.0);
    }

    frame.peak = peak;
    frame.rms = std::sqrt(energy / m_windowSize);

    FftHelper::fft(buffer);

    int bins = m_windowSize / 2;
    frame.magnitude.resize(bins);

    double maxMag = 1e-12;
    int maxBin = 0;

    for (int i = 0; i < bins; ++i) {
        double mag = std::abs(buffer[i]);
        frame.magnitude[i] = mag;

        if (mag > maxMag) {
            maxMag = mag;
            maxBin = i;
        }
    }

    for (int i = 0; i < bins; ++i) {
        frame.magnitude[i] /= maxMag;
    }

    frame.dominantFrequency =
        static_cast<double>(maxBin) * m_sampleRate / m_windowSize;

    return frame;
}