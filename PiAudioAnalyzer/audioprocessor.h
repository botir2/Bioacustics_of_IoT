#ifndef AUDIOPROCESSOR_H
#define AUDIOPROCESSOR_H

#include <vector>
#include <complex>
#include <QString>
#include <QPixmap>
#include <QVector>

struct AnalysisResult
{
    bool success = false;
    QString message;

    QPixmap waveformPixmap;
    QPixmap fftPixmap;
    QPixmap spectrogramPixmap;

    int sampleRate = 0;
    int sampleCount = 0;
    double duration = 0.0;
};

class AudioProcessor
{
public:
    static AnalysisResult analyse(const QString &filePath);

private:
    static bool readWav16(const QString &filePath,
                          QVector<double> &samples,
                          int &sampleRate,
                          QString &errorMessage);

    static void fft(std::vector<std::complex<double>> &a);

    static QVector<double> calculateSpectrum(const QVector<double> &samples,
                                             int fftSize);

    static QPixmap drawWaveform(const QVector<double> &samples,
                                int sampleRate);

    static QPixmap drawSpectrum(const QVector<double> &spectrum,
                                int sampleRate,
                                int fftSize);

    static QPixmap drawSpectrogram(const QVector<double> &samples,
                                   int sampleRate);
};

#endif
