#ifndef FEATUREMAPRENDERER_H
#define FEATUREMAPRENDERER_H

#include <QColor>
#include <QImage>
#include <QPixmap>
#include <QSize>
#include <QVector>

enum class FeatureMapColourStyle
{
    MelGreenYellow,
    LogMelHot,
    MfccDiverging
};

class FeatureMapRenderer
{
public:
    static QImage createImage(QSize size);

    static void appendColumn(QImage &image,
                             const QVector<double> &values,
                             FeatureMapColourStyle style);

    static QPixmap toPixmap(const QImage &image);

private:
    static double normaliseSequential(double value,
                                      double minValue,
                                      double maxValue);

    static double normaliseDiverging(double value,
                                     double maxAbsValue);

    static QColor colourForValue(double normalisedValue,
                                 FeatureMapColourStyle style);

    static QColor interpolate(const QColor &a,
                              const QColor &b,
                              double t);
};

#endif