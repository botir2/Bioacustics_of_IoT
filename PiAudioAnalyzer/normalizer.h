#ifndef NORMALIZER_H
#define NORMALIZER_H

#include <QVector>
#include <QString>

class Normalizer
{
public:
    static QVector<double> apply(const QVector<double>& samples, const QString& method);

private:
    static QVector<double> peakNormalize(const QVector<double>& samples);
    static QVector<double> zScoreNormalize(const QVector<double>& samples);
    static QVector<double> minMaxNormalize(const QVector<double>& samples);
    static QVector<double> rmsNormalize(const QVector<double>& samples);
    static QVector<double> meanCenter(const QVector<double>& samples);
};

#endif // NORMALIZER_H