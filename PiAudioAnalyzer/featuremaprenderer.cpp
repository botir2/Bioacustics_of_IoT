#include "featuremaprenderer.h"

#include <QPainter>
#include <QtGlobal>

#include <algorithm>
#include <cmath>

QImage FeatureMapRenderer::createImage(QSize size)
{
    if (size.width() < 50 || size.height() < 50) {
        size = QSize(900, 320);
    }

    QImage image(size, QImage::Format_RGB32);
    image.fill(QColor("#020617"));

    return image;
}

void FeatureMapRenderer::appendColumn(QImage &image,
                                      const QVector<double> &values,
                                      FeatureMapColourStyle style)
{
    if (image.isNull() || values.isEmpty()) {
        return;
    }

    // Shift previous heatmap image one pixel to the left.
    QImage shifted = image.copy(1, 0, image.width() - 1, image.height());

    {
        QPainter painter(&image);
        painter.drawImage(0, 0, shifted);
    }

    int valueCount = static_cast<int>(values.size());

    double minValue = values[0];
    double maxValue = values[0];
    double maxAbsValue = std::abs(values[0]);

    for (double v : values) {
        minValue = std::min(minValue, v);
        maxValue = std::max(maxValue, v);
        maxAbsValue = std::max(maxAbsValue, std::abs(v));
    }

    if (maxAbsValue < 1e-12) {
        maxAbsValue = 1e-12;
    }

    int x = image.width() - 1;

    for (int y = 0; y < image.height(); ++y) {
        int index = static_cast<int>(
            static_cast<double>(image.height() - 1 - y) /
            image.height() * (valueCount - 1)
        );

        index = qBound(0, index, valueCount - 1);

        double normalised = 0.0;

        if (style == FeatureMapColourStyle::MfccDiverging) {
            normalised = normaliseDiverging(values[index], maxAbsValue);
        } else {
            normalised = normaliseSequential(values[index], minValue, maxValue);
        }

        QColor colour = colourForValue(normalised, style);
        image.setPixelColor(x, y, colour);
    }
}

QPixmap FeatureMapRenderer::toPixmap(const QImage &image)
{
    return QPixmap::fromImage(image);
}

double FeatureMapRenderer::normaliseSequential(double value,
                                               double minValue,
                                               double maxValue)
{
    double range = maxValue - minValue;

    if (std::abs(range) < 1e-12) {
        return 0.0;
    }

    double out = (value - minValue) / range;
    return qBound(0.0, out, 1.0);
}

double FeatureMapRenderer::normaliseDiverging(double value,
                                              double maxAbsValue)
{
    // 0.0 = strong negative
    // 0.5 = near zero
    // 1.0 = strong positive
    double out = 0.5 + (value / (2.0 * maxAbsValue));
    return qBound(0.0, out, 1.0);
}

QColor FeatureMapRenderer::colourForValue(double normalisedValue,
                                          FeatureMapColourStyle style)
{
    double value = qBound(0.0, normalisedValue, 1.0);

    if (style == FeatureMapColourStyle::MelGreenYellow) {
        // Mel Spectrogram:
        // dark navy -> green -> yellow -> pale yellow
        QColor c0("#020617");
        QColor c1("#064E3B");
        QColor c2("#22C55E");
        QColor c3("#FACC15");
        QColor c4("#FFF7AE");

        if (value < 0.25) {
            return interpolate(c0, c1, value / 0.25);
        } else if (value < 0.50) {
            return interpolate(c1, c2, (value - 0.25) / 0.25);
        } else if (value < 0.75) {
            return interpolate(c2, c3, (value - 0.50) / 0.25);
        } else {
            return interpolate(c3, c4, (value - 0.75) / 0.25);
        }
    }

    if (style == FeatureMapColourStyle::LogMelHot) {
        // Log-Mel Spectrogram:
        // black -> purple/red -> orange -> yellow -> pale yellow
        QColor c0("#050505");
        QColor c1("#3B0A45");
        QColor c2("#9D174D");
        QColor c3("#F97316");
        QColor c4("#FACC15");
        QColor c5("#FFF7AE");

        if (value < 0.20) {
            return interpolate(c0, c1, value / 0.20);
        } else if (value < 0.40) {
            return interpolate(c1, c2, (value - 0.20) / 0.20);
        } else if (value < 0.60) {
            return interpolate(c2, c3, (value - 0.40) / 0.20);
        } else if (value < 0.80) {
            return interpolate(c3, c4, (value - 0.60) / 0.20);
        } else {
            return interpolate(c4, c5, (value - 0.80) / 0.20);
        }
    }

    // MFCC:
    // blue = negative coefficient
    // dark = near zero
    // orange/yellow = positive coefficient
    QColor c0("#1E3A8A");
    QColor c1("#2563EB");
    QColor c2("#111827");
    QColor c3("#F97316");
    QColor c4("#FACC15");

    if (value < 0.25) {
        return interpolate(c0, c1, value / 0.25);
    } else if (value < 0.50) {
        return interpolate(c1, c2, (value - 0.25) / 0.25);
    } else if (value < 0.75) {
        return interpolate(c2, c3, (value - 0.50) / 0.25);
    } else {
        return interpolate(c3, c4, (value - 0.75) / 0.25);
    }
}

QColor FeatureMapRenderer::interpolate(const QColor &a,
                                        const QColor &b,
                                        double t)
{
    t = qBound(0.0, t, 1.0);

    int red = static_cast<int>(a.red() + (b.red() - a.red()) * t);
    int green = static_cast<int>(a.green() + (b.green() - a.green()) * t);
    int blue = static_cast<int>(a.blue() + (b.blue() - a.blue()) * t);

    return QColor(red, green, blue);
}