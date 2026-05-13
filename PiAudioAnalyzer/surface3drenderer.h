#ifndef SURFACE3DRENDERER_H
#define SURFACE3DRENDERER_H

#include <QColor>
#include <QPixmap>
#include <QPointF>
#include <QRectF>
#include <QSize>
#include <QVector>

#include "stftprocessor.h"

class Surface3DRenderer
{
public:
    void appendFrames(const QVector<StftFrame> &frames,
                      int maxHistoryFrames = 120);

    void reset();

    QPixmap renderSurface(QSize size,
                          int sampleRate,
                          int maxFrequencyHz = 8000) const;

private:
    QVector<QVector<double>> history;

    static double toDb(double value);
    static QColor colourFromDb(double dbValue);

    static QPointF projectPoint(double x,
                                double y,
                                double z,
                                const QRectF &plotRect);
};

#endif
