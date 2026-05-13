#include "realtimefeaturehelper.h"

#include "melspec.h"
#include "logmel.h"
#include "mfccprocessor.h"
#include "featuremaprenderer.h"

void RealtimeFeatureHelper::updateMelImage(QImage &image,
                                           const QVector<StftFrame> &frames,
                                           int sampleRate,
                                           int fftSize,
                                           int melFilters,
                                           int minFreq,
                                           int maxFreq,
                                           bool usePower)
{
    for (const StftFrame &frame : frames) {
        QVector<double> melValues =
            MelSpec::compute(
                frame.magnitude,
                sampleRate,
                fftSize,
                melFilters,
                minFreq,
                maxFreq,
                usePower
            );

        FeatureMapRenderer::appendColumn(
            image,
            melValues,
            FeatureMapColourStyle::MelGreenYellow
        );
    }
}

void RealtimeFeatureHelper::updateLogMelImage(QImage &image,
                                              const QVector<StftFrame> &frames,
                                              int sampleRate,
                                              int fftSize,
                                              int melFilters,
                                              int minFreq,
                                              int maxFreq,
                                              double epsilon,
                                              bool useDb)
{
    for (const StftFrame &frame : frames) {
        QVector<double> logMelValues =
            LogMel::compute(
                frame.magnitude,
                sampleRate,
                fftSize,
                melFilters,
                minFreq,
                maxFreq,
                epsilon,
                useDb
            );

        FeatureMapRenderer::appendColumn(
            image,
            logMelValues,
            FeatureMapColourStyle::LogMelHot
        );
    }
}

QVector<double> RealtimeFeatureHelper::updateMfccImage(QImage &image,
                                                       const QVector<StftFrame> &frames,
                                                       int sampleRate,
                                                       int fftSize,
                                                       int melFilters,
                                                       int coeffCount,
                                                       int minFreq,
                                                       int maxFreq,
                                                       bool includeC0)
{
    QVector<double> lastMfccValues;

    for (const StftFrame &frame : frames) {
        QVector<double> mfccValues =
            MfccProcessor::compute(
                frame.magnitude,
                sampleRate,
                fftSize,
                melFilters,
                coeffCount,
                minFreq,
                maxFreq,
                includeC0
            );

        FeatureMapRenderer::appendColumn(
            image,
            mfccValues,
            FeatureMapColourStyle::MfccDiverging
        );

        lastMfccValues = mfccValues;
    }

    return lastMfccValues;
}