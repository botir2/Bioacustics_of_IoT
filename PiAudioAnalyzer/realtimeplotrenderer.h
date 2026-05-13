#ifndef REALTIMEPLOTRENDERER_H
#define REALTIMEPLOTRENDERER_H

#include <QImage>
#include <QPixmap>
#include <QSize>
#include <QVector>

class RealtimePlotRenderer
{
public:
    static QPixmap drawWaveform(const QVector<double> &samples, QSize size);

    static QImage createSpectrogramImage(QSize size);

    static void appendSpectrogramColumn(QImage &image,
                                        const QVector<double> &magnitude);

    static QPixmap toPixmap(const QImage &image);
};

#endif