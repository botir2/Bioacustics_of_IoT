#include "noisereducer.h"
#include <QtMath>

NoiseReducer::NoiseReducer()
{
}

QVector<float> NoiseReducer::reduce(const QVector<float>& inputSamples,
                                    float threshold,
                                    float strength,
                                    float smoothing,
                                    bool nonStationary)
{
    QVector<float> output;
    output.reserve(inputSamples.size());

    float previousGain = 1.0f;

    for (float sample : inputSamples) {
        float absSample = qAbs(sample);

        float targetGain = 1.0f;

        // Stationary mode: simple fixed noise gate
        if (!nonStationary) {
            if (absSample < threshold) {
                targetGain = 1.0f - strength;
            }
        }

        // Non-stationary mode: softer adaptive reduction
        else {
            float adaptiveThreshold = threshold * 1.3f;

            if (absSample < adaptiveThreshold) {
                float ratio = absSample / adaptiveThreshold;
                targetGain = (1.0f - strength) + (strength * ratio);
            }
        }

        // Smoothing: prevents sudden harsh changes
        float smoothAmount = qBound(0.0f, smoothing, 0.95f);
        float gain = smoothAmount * previousGain + (1.0f - smoothAmount) * targetGain;

        output.append(sample * gain);
        previousGain = gain;
    }

    return output;
}
