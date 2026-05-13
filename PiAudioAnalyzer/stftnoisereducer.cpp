#include "stftnoisereducer.h"

#include <algorithm>
#include <cmath>

StftNoiseReducer::StftNoiseReducer()
{
}

void StftNoiseReducer::reset()
{
    noiseFloor.clear();
    previousOutput.clear();
}

void StftNoiseReducer::ensureSize(int size)
{
    if (noiseFloor.size() != size) {
        noiseFloor = QVector<double>(size, 0.0);
    }

    if (previousOutput.size() != size) {
        previousOutput = QVector<double>(size, 0.0);
    }
}

double StftNoiseReducer::estimateFrameNoise(const QVector<double> &magnitude) const
{
    if (magnitude.isEmpty()) {
        return 0.0;
    }

    QVector<double> sorted = magnitude;
    std::sort(sorted.begin(), sorted.end());

    // Use the 20th percentile as a simple noise-floor estimate.
    const int size = static_cast<int>(sorted.size());

    int index = static_cast<int>(size * 0.20);

    if (index < 0) {
        index = 0;
    }

    if (index > size - 1) {
        index = size - 1;
    }

    return sorted[index];
}

QVector<double> StftNoiseReducer::processMagnitude(const QVector<double> &magnitude,
                                                   double threshold,
                                                   double strength,
                                                   double smoothing,
                                                   bool nonStationary)
{
    if (magnitude.isEmpty()) {
        return {};
    }

    ensureSize(magnitude.size());

    threshold = std::max(0.0, std::min(threshold, 1.0));
    strength = std::max(0.0, std::min(strength, 1.0));
    smoothing = std::max(0.0, std::min(smoothing, 1.0));

    QVector<double> output;
    output.resize(magnitude.size());

    double frameNoise = estimateFrameNoise(magnitude);

    for (int i = 0; i < magnitude.size(); ++i) {
        double current = magnitude[i];

        if (noiseFloor[i] <= 0.0) {
            noiseFloor[i] = frameNoise;
        }

        // Stationary mode updates slowly.
        // Non-stationary mode updates faster.
        double updateRate = nonStationary ? 0.08 : 0.015;
        noiseFloor[i] = (1.0 - updateRate) * noiseFloor[i] + updateRate * frameNoise;

        double gateLevel = noiseFloor[i] + threshold;

        double reduced = current;

        if (current < gateLevel) {
            // Reduce low-energy bins.
            reduced = current * (1.0 - strength);
        } else {
            // Keep stronger signal bins.
            reduced = current;
        }

        // Smooth output over time to reduce flicker.
        double smoothed =
            smoothing * previousOutput[i] +
            (1.0 - smoothing) * reduced;

        output[i] = smoothed;
        previousOutput[i] = smoothed;
    }

    return output;
}
