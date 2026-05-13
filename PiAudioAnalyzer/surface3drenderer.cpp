#include "surface3drenderer.h"

#include <QPainter>
#include <QPainterPath>
#include <QPen>
#include <QBrush>
#include <QColor>
#include <QtGlobal>

#include <algorithm>
#include <cmath>

void Surface3DRenderer::appendFrames(const QVector<StftFrame> &frames,
                                     int maxHistoryFrames)
{
    for (const StftFrame &frame : frames) {
        if (!frame.magnitude.isEmpty()) {
            history.append(frame.magnitude);
        }
    }

    while (history.size() > maxHistoryFrames) {
        history.removeFirst();
    }
}

void Surface3DRenderer::reset()
{
    history.clear();
}

QPixmap Surface3DRenderer::renderSurface(QSize size,
                                         int sampleRate,
                                         int maxFrequencyHz) const
{
    if (size.width() < 100 || size.height() < 100) {
        size = QSize(800, 520);
    }

    QPixmap pixmap(size);
    pixmap.fill(QColor("#020617"));

    QPainter painter(&pixmap);
    painter.setRenderHint(QPainter::Antialiasing);

    QRectF plotRect(50, 35, size.width() - 100, size.height() - 90);

    // Title
    painter.setPen(QColor("#22D3EE"));
    painter.setFont(QFont("Arial", 11, QFont::Bold));
    painter.drawText(20, 24, "3D STFT Spectrogram Surface");

    // Empty state
    if (history.isEmpty()) {
        painter.setPen(QColor("#CBD5E1"));
        painter.drawText(plotRect, Qt::AlignCenter,
                         "Waiting for real-time STFT frames...\n"
                         "X = Time, Y = Frequency, Z = Magnitude / dB");
        return pixmap;
    }

    int timeCount = history.size();
    int binCount = history.last().size();

    if (binCount <= 1) {
        return pixmap;
    }

    int nyquist = sampleRate / 2;
    maxFrequencyHz = std::min(maxFrequencyHz, nyquist);

    int maxBin = static_cast<int>(
        static_cast<double>(maxFrequencyHz) / nyquist * (binCount - 1)
        );

    maxBin = qBound(1, maxBin, binCount - 1);

    // Draw 3D box/grid.
    painter.setPen(QPen(QColor("#334155"), 1));

    QPointF backLeft = plotRect.topLeft() + QPointF(120, 40);
    QPointF frontLeft = plotRect.bottomLeft() + QPointF(40, -30);
    QPointF frontRight = plotRect.bottomRight() + QPointF(-70, -10);
    QPointF backRight = plotRect.topRight() + QPointF(-120, 60);

    painter.drawLine(frontLeft, frontRight);
    painter.drawLine(frontLeft, backLeft);
    painter.drawLine(frontRight, backRight);
    painter.drawLine(backLeft, backRight);

    for (int i = 0; i <= 5; ++i) {
        double t = static_cast<double>(i) / 5.0;

        QPointF a = frontLeft * (1.0 - t) + frontRight * t;
        QPointF b = backLeft * (1.0 - t) + backRight * t;
        painter.drawLine(a, b);

        QPointF c = frontLeft * (1.0 - t) + backLeft * t;
        QPointF d = frontRight * (1.0 - t) + backRight * t;
        painter.drawLine(c, d);
    }

    // Render surface from back to front.
    // Each small quad is coloured by average dB magnitude.
    for (int t = 0; t < timeCount - 1; ++t) {
        for (int b = 0; b < maxBin - 1; ++b) {
            double v00 = history[t][b];
            double v01 = history[t][b + 1];
            double v10 = history[t + 1][b];
            double v11 = history[t + 1][b + 1];

            double db = (toDb(v00) + toDb(v01) + toDb(v10) + toDb(v11)) / 4.0;

            double x0 = static_cast<double>(t) / (timeCount - 1);
            double x1 = static_cast<double>(t + 1) / (timeCount - 1);

            double y0 = static_cast<double>(b) / maxBin;
            double y1 = static_cast<double>(b + 1) / maxBin;

            double z00 = (toDb(v00) + 100.0) / 100.0;
            double z01 = (toDb(v01) + 100.0) / 100.0;
            double z10 = (toDb(v10) + 100.0) / 100.0;
            double z11 = (toDb(v11) + 100.0) / 100.0;

            z00 = qBound(0.0, z00, 1.0);
            z01 = qBound(0.0, z01, 1.0);
            z10 = qBound(0.0, z10, 1.0);
            z11 = qBound(0.0, z11, 1.0);

            QPolygonF poly;
            poly << projectPoint(x0, y0, z00, plotRect)
                 << projectPoint(x1, y0, z10, plotRect)
                 << projectPoint(x1, y1, z11, plotRect)
                 << projectPoint(x0, y1, z01, plotRect);

            QColor colour = colourFromDb(db);
            painter.setPen(Qt::NoPen);
            painter.setBrush(colour);
            painter.drawPolygon(poly);
        }
    }

    // Axes labels
    painter.setPen(QColor("#E5E7EB"));
    painter.setFont(QFont("Arial", 9));

    painter.drawText(QPointF(plotRect.right() - 80, plotRect.bottom() - 5), "Time");
    painter.drawText(QPointF(plotRect.left() + 10, plotRect.bottom() - 5), "0");
    painter.drawText(QPointF(plotRect.left() + 35, plotRect.top() + 25), "Magnitude (dB)");
    painter.drawText(QPointF(plotRect.left() + 75, plotRect.bottom() - 40), "Frequency");

    painter.setPen(QColor("#93C5FD"));
    painter.drawText(QPointF(plotRect.right() - 160, plotRect.top() + 25),
                     "Max freq: " + QString::number(maxFrequencyHz) + " Hz");

    // Colour scale
    int scaleX = size.width() - 45;
    int scaleY = 75;
    int scaleH = 300;

    for (int i = 0; i < scaleH; ++i) {
        double ratio = 1.0 - static_cast<double>(i) / scaleH;
        double db = -100.0 + ratio * 100.0;
        painter.setPen(colourFromDb(db));
        painter.drawLine(scaleX, scaleY + i, scaleX + 18, scaleY + i);
    }

    painter.setPen(QColor("#E5E7EB"));
    painter.drawText(scaleX - 2, scaleY - 10, "dB");
    painter.drawText(scaleX + 25, scaleY + 5, "0");
    painter.drawText(scaleX + 25, scaleY + scaleH / 2, "-50");
    painter.drawText(scaleX + 25, scaleY + scaleH, "-100");

    return pixmap;
}

double Surface3DRenderer::toDb(double value)
{
    double safe = std::max(value, 1e-12);
    double db = 20.0 * std::log10(safe);

    // Clamp to visible range.
    return qBound(-100.0, db, 0.0);
}

QColor Surface3DRenderer::colourFromDb(double dbValue)
{
    // Convert -100..0 dB to 0..1.
    double t = (dbValue + 100.0) / 100.0;
    t = qBound(0.0, t, 1.0);

    // Inferno/hot-like palette:
    QColor c0("#12002B"); // dark purple
    QColor c1("#3B0764"); // purple
    QColor c2("#BE123C"); // red
    QColor c3("#F97316"); // orange
    QColor c4("#FACC15"); // yellow

    auto mix = [](const QColor &a, const QColor &b, double r) {
        r = qBound(0.0, r, 1.0);
        int red = static_cast<int>(a.red() + (b.red() - a.red()) * r);
        int green = static_cast<int>(a.green() + (b.green() - a.green()) * r);
        int blue = static_cast<int>(a.blue() + (b.blue() - a.blue()) * r);
        return QColor(red, green, blue);
    };

    if (t < 0.25) {
        return mix(c0, c1, t / 0.25);
    }

    if (t < 0.50) {
        return mix(c1, c2, (t - 0.25) / 0.25);
    }

    if (t < 0.75) {
        return mix(c2, c3, (t - 0.50) / 0.25);
    }

    return mix(c3, c4, (t - 0.75) / 0.25);
}

QPointF Surface3DRenderer::projectPoint(double x,
                                        double y,
                                        double z,
                                        const QRectF &plotRect)
{
    // Pseudo-3D projection:
    // x = time, y = frequency, z = magnitude height.
    double baseX = plotRect.left() + 140 + x * (plotRect.width() - 260);
    double baseY = plotRect.bottom() - 45 - y * (plotRect.height() - 140);

    // Frequency depth slants backwards.
    double depthX = y * 150.0;
    double depthY = y * 70.0;

    // Magnitude lifts the point upward.
    double height = z * 170.0;

    return QPointF(baseX + depthX, baseY - depthY - height);
}
