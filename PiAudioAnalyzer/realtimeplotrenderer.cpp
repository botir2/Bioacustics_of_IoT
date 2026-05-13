#include "realtimeplotrenderer.h"

#include <QColor>
#include <QPainter>
#include <QPainterPath>
#include <QPen>
#include <QtGlobal>

#include <cmath>

QPixmap RealtimePlotRenderer::drawWaveform(const QVector<double> &samples, QSize size)
{
    if (size.width() < 50 || size.height() < 50) {
        size = QSize(900, 150);
    }

    QPixmap pixmap(size);
    pixmap.fill(Qt::white);

    QPainter painter(&pixmap);
    painter.setRenderHint(QPainter::Antialiasing);

    QRect graphRect(40, 15, size.width() - 55, size.height() - 35);

    painter.setPen(Qt::black);
    painter.drawRect(graphRect);
    painter.drawText(8, 12, "Live waveform");

    if (samples.isEmpty()) {
        return pixmap;
    }

    int midY = graphRect.center().y();

    painter.setPen(QPen(Qt::gray, 1, Qt::DashLine));
    painter.drawLine(graphRect.left(), midY, graphRect.right(), midY);

    QPainterPath path;

    for (int x = 0; x < graphRect.width(); ++x) {
        int index = static_cast<int>(
            static_cast<double>(x) / graphRect.width() * (samples.size() - 1)
        );

        double value = samples[index];
        int y = midY - static_cast<int>(value * graphRect.height() * 0.45);

        if (x == 0) {
            path.moveTo(graphRect.left() + x, y);
        } else {
            path.lineTo(graphRect.left() + x, y);
        }
    }

    painter.setPen(QPen(Qt::blue, 1));
    painter.drawPath(path);

    return pixmap;
}

QImage RealtimePlotRenderer::createSpectrogramImage(QSize size)
{
    if (size.width() < 50 || size.height() < 50) {
        size = QSize(900, 280);
    }

    QImage image(size, QImage::Format_RGB32);
    image.fill(Qt::black);

    return image;
}

void RealtimePlotRenderer::appendSpectrogramColumn(QImage &image,
                                                   const QVector<double> &magnitude)
{
    if (image.isNull() || magnitude.isEmpty()) {
        return;
    }

    QImage shifted = image.copy(1, 0, image.width() - 1, image.height());

    {
        QPainter painter(&image);
        painter.drawImage(0, 0, shifted);
    }

    int x = image.width() - 1;

    for (int y = 0; y < image.height(); ++y) {
        int bin = static_cast<int>(
            static_cast<double>(image.height() - 1 - y) /
            image.height() * (magnitude.size() - 1)
        );

        double value = magnitude[bin];

        value = std::log10(1.0 + 50.0 * value) / std::log10(51.0);

        int shade = qBound(0, static_cast<int>(value * 255.0), 255);

        image.setPixel(x, y, QColor(shade, shade, shade).rgb());
    }
}

QPixmap RealtimePlotRenderer::toPixmap(const QImage &image)
{
    return QPixmap::fromImage(image);
}