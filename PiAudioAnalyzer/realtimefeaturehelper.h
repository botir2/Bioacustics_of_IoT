#ifndef REALTIMEFEATUREHELPER_H
#define REALTIMEFEATUREHELPER_H

#include <QImage>
#include <QVector>

#include "stftprocessor.h"

class RealtimeFeatureHelper
{
public:
    static void updateMelImage(QImage &image,
                               const QVector<StftFrame> &frames,
                               int sampleRate,
                               int fftSize,
                               int melFilters,
                               int minFreq,
                               int maxFreq,
                               bool usePower);

    static void updateLogMelImage(QImage &image,
                                  const QVector<StftFrame> &frames,
                                  int sampleRate,
                                  int fftSize,
                                  int melFilters,
                                  int minFreq,
                                  int maxFreq,
                                  double epsilon,
                                  bool useDb);

    static QVector<double> updateMfccImage(QImage &image,
                                           const QVector<StftFrame> &frames,
                                           int sampleRate,
                                           int fftSize,
                                           int melFilters,
                                           int coeffCount,
                                           int minFreq,
                                           int maxFreq,
                                           bool includeC0);
};

#endif