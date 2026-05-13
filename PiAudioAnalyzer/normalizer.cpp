#include "normalizer.h"
#include <cmath>
#include <algorithm>

QVector<double> Normalizer::apply(const QVector<double>& samples, const QString& method)
{
    if (method == "Peak normalisation")
        return peakNormalize(samples);

    if (method == "Z-score normalisation")
        return zScoreNormalize(samples);

    if (method == "Min-max normalisation")
        return minMaxNormalize(samples);

    if (method == "RMS normalisation")
        return rmsNormalize(samples);

    if (method == "Mean centring")
        return meanCenter(samples);

    return samples;
}

QVector<double> Normalizer::peakNormalize(const QVector<double>& samples)
{
    QVector<double> y = samples;
    const double eps = 1e-12;

    double peak = 0.0;
    for (double v : samples)
        peak = std::max(peak, std::abs(v));

    for (double& v : y)
        v = v / (peak + eps);

    return y;
}

QVector<double> Normalizer::zScoreNormalize(const QVector<double>& samples)
{
    QVector<double> y = samples;
    const double eps = 1e-12;

    if (samples.isEmpty())
        return y;

    double mean = 0.0;
    for (double v : samples)
        mean += v;
    mean /= samples.size();

    double var = 0.0;
    for (double v : samples)
        var += (v - mean) * (v - mean);
    var /= samples.size();

    double stddev = std::sqrt(var + eps);

    for (double& v : y)
        v = (v - mean) / stddev;

    return y;
}

QVector<double> Normalizer::minMaxNormalize(const QVector<double>& samples)
{
    QVector<double> y = samples;
    const double eps = 1e-12;

    if (samples.isEmpty())
        return y;

    double minVal = samples[0];
    double maxVal = samples[0];

    for (double v : samples) {
        minVal = std::min(minVal, v);
        maxVal = std::max(maxVal, v);
    }

    for (double& v : y)
        v = (v - minVal) / (maxVal - minVal + eps);

    return y;
}

QVector<double> Normalizer::rmsNormalize(const QVector<double>& samples)
{
    QVector<double> y = samples;
    const double eps = 1e-12;

    if (samples.isEmpty())
        return y;

    double rms = 0.0;
    for (double v : samples)
        rms += v * v;

    rms = std::sqrt(rms / samples.size() + eps);

    for (double& v : y)
        v = v / rms;

    return y;
}

QVector<double> Normalizer::meanCenter(const QVector<double>& samples)
{
    QVector<double> y = samples;

    if (samples.isEmpty())
        return y;

    double mean = 0.0;
    for (double v : samples)
        mean += v;
    mean /= samples.size();

    for (double& v : y)
        v = v - mean;

    return y;
}