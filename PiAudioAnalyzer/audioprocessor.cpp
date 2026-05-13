#include "audioprocessor.h"

#include <QFile>
#include <QPainter>
#include <QPainterPath>
#include <QImage>
#include <QColor>
#include <QPen>
#include <QRect>
#include <QtGlobal>

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

AnalysisResult AudioProcessor::analyse(const QString &filePath)
{
    AnalysisResult result;

    QVector<double> samples;
    int sampleRate = 0;
    QString errorMessage;

    if (!readWav16(filePath, samples, sampleRate, errorMessage)) {
        result.success = false;
        result.message = errorMessage;
        return result;
    }

    if (samples.isEmpty()) {
        result.success = false;
        result.message = "The WAV file contains no audio samples.";
        return result;
    }

    int fftSize = 8192;

    if (samples.size() < fftSize)
        fftSize = 1024;

    QVector<double> spectrum = calculateSpectrum(samples, fftSize);

    result.waveformPixmap = drawWaveform(samples, sampleRate);
    result.fftPixmap = drawSpectrum(spectrum, sampleRate, fftSize);
    result.spectrogramPixmap = drawSpectrogram(samples, sampleRate);

    result.sampleRate = sampleRate;
    result.sampleCount = samples.size();
    result.duration = static_cast<double>(samples.size()) / sampleRate;
    result.success = true;

    result.message =
        "Analysis completed.\n\n"
        "File: " + filePath + "\n"
                     "Sample rate: " + QString::number(sampleRate) + " Hz\n"
                                        "Samples: " + QString::number(samples.size()) + "\n"
                                            "Duration: " + QString::number(result.duration, 'f', 2) + " seconds\n"
                                                     "FFT size: " + QString::number(fftSize) + "\n\n"
                                     "Generated outputs:\n"
                                     "- Waveform\n"
                                     "- FFT magnitude spectrum\n"
                                     "- Spectrogram";

    return result;
}

bool AudioProcessor::readWav16(const QString &filePath,
                               QVector<double> &samples,
                               int &sampleRate,
                               QString &errorMessage)
{
    QFile file(filePath);

    if (!file.open(QIODevice::ReadOnly)) {
        errorMessage = "Cannot open WAV file:\n" + filePath;
        return false;
    }

    QByteArray bytes = file.readAll();
    file.close();

    if (bytes.size() < 44) {
        errorMessage = "File is too small to be a valid WAV file.";
        return false;
    }

    auto readU16 = [&](int pos) -> quint16 {
        return static_cast<quint8>(bytes[pos]) |
               (static_cast<quint16>(static_cast<quint8>(bytes[pos + 1])) << 8);
    };

    auto readU32 = [&](int pos) -> quint32 {
        return static_cast<quint8>(bytes[pos]) |
               (static_cast<quint32>(static_cast<quint8>(bytes[pos + 1])) << 8) |
               (static_cast<quint32>(static_cast<quint8>(bytes[pos + 2])) << 16) |
               (static_cast<quint32>(static_cast<quint8>(bytes[pos + 3])) << 24);
    };

    if (bytes.mid(0, 4) != "RIFF" || bytes.mid(8, 4) != "WAVE") {
        errorMessage = "This file is not a valid RIFF/WAVE file.";
        return false;
    }

    int audioFormat = 0;
    int channels = 0;
    int bitsPerSample = 0;
    int dataOffset = -1;
    int dataSize = 0;
    sampleRate = 0;

    int pos = 12;

    while (pos + 8 <= bytes.size()) {
        QByteArray chunkId = bytes.mid(pos, 4);
        quint32 chunkSize = readU32(pos + 4);
        int chunkDataStart = pos + 8;

        if (chunkDataStart + static_cast<int>(chunkSize) > bytes.size())
            break;

        if (chunkId == "fmt ") {
            if (chunkSize < 16) {
                errorMessage = "Invalid WAV fmt chunk.";
                return false;
            }

            audioFormat = readU16(chunkDataStart + 0);
            channels = readU16(chunkDataStart + 2);
            sampleRate = static_cast<int>(readU32(chunkDataStart + 4));
            bitsPerSample = readU16(chunkDataStart + 14);
        }
        else if (chunkId == "data") {
            dataOffset = chunkDataStart;
            dataSize = static_cast<int>(chunkSize);
        }

        pos = chunkDataStart + static_cast<int>(chunkSize);

        if (chunkSize % 2 == 1)
            pos++;
    }

    if (audioFormat != 1) {
        errorMessage = "Only uncompressed PCM WAV files are supported.";
        return false;
    }

    if (bitsPerSample != 16) {
        errorMessage = "Only 16-bit PCM WAV files are supported in this version.";
        return false;
    }

    if (channels < 1) {
        errorMessage = "Invalid number of audio channels.";
        return false;
    }

    if (sampleRate <= 0) {
        errorMessage = "Invalid sample rate.";
        return false;
    }

    if (dataOffset < 0 || dataSize <= 0) {
        errorMessage = "No WAV data chunk found.";
        return false;
    }

    int bytesPerSample = bitsPerSample / 8;
    int frameSize = channels * bytesPerSample;
    int frameCount = dataSize / frameSize;

    samples.clear();
    samples.reserve(frameCount);

    for (int i = 0; i < frameCount; ++i) {
        double monoSum = 0.0;

        for (int ch = 0; ch < channels; ++ch) {
            int samplePos = dataOffset + i * frameSize + ch * bytesPerSample;

            quint16 raw = readU16(samplePos);
            qint16 signedSample = static_cast<qint16>(raw);

            monoSum += static_cast<double>(signedSample) / 32768.0;
        }

        samples.append(monoSum / channels);
    }

    return true;
}

void AudioProcessor::fft(std::vector<std::complex<double>> &a)
{
    int n = static_cast<int>(a.size());

    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;

        for (; j & bit; bit >>= 1)
            j ^= bit;

        j ^= bit;

        if (i < j)
            std::swap(a[i], a[j]);
    }

    const double pi = std::acos(-1.0);

    for (int len = 2; len <= n; len <<= 1) {
        double angle = -2.0 * pi / len;
        std::complex<double> wlen(std::cos(angle), std::sin(angle));

        for (int i = 0; i < n; i += len) {
            std::complex<double> w(1.0, 0.0);

            for (int j = 0; j < len / 2; j++) {
                std::complex<double> u = a[i + j];
                std::complex<double> v = a[i + j + len / 2] * w;

                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;

                w *= wlen;
            }
        }
    }
}

QVector<double> AudioProcessor::calculateSpectrum(const QVector<double> &samples,
                                                  int fftSize)
{
    std::vector<std::complex<double>> buffer(fftSize);
    int copyCount = std::min(static_cast<int>(samples.size()), fftSize);

    const double pi = std::acos(-1.0);

    for (int i = 0; i < copyCount; ++i) {
        double hann = 0.5 * (1.0 - std::cos(2.0 * pi * i / (fftSize - 1)));
        buffer[i] = std::complex<double>(samples[i] * hann, 0.0);
    }

    for (int i = copyCount; i < fftSize; ++i)
        buffer[i] = std::complex<double>(0.0, 0.0);

    fft(buffer);

    QVector<double> magnitude;
    magnitude.resize(fftSize / 2);

    double maxValue = 1e-12;

    for (int i = 0; i < fftSize / 2; ++i) {
        magnitude[i] = std::abs(buffer[i]);

        if (magnitude[i] > maxValue)
            maxValue = magnitude[i];
    }

    for (int i = 0; i < magnitude.size(); ++i)
        magnitude[i] /= maxValue;

    return magnitude;
}

QPixmap AudioProcessor::drawWaveform(const QVector<double> &samples,
                                     int sampleRate)
{
    QSize size(900, 260);
    QPixmap pixmap(size);
    pixmap.fill(Qt::white);

    QPainter p(&pixmap);
    p.setRenderHint(QPainter::Antialiasing);

    QRect graphRect(55, 25, size.width() - 70, size.height() - 60);

    p.setPen(Qt::black);
    p.drawRect(graphRect);
    p.drawText(10, 18, "Waveform");

    if (samples.isEmpty())
        return pixmap;

    int midY = graphRect.center().y();

    p.setPen(QPen(Qt::gray, 1, Qt::DashLine));
    p.drawLine(graphRect.left(), midY, graphRect.right(), midY);

    QPainterPath path;

    for (int x = 0; x < graphRect.width(); ++x) {
        int index = 0;

        if (samples.size() > 1) {
            index = static_cast<int>(
                static_cast<double>(x) / graphRect.width() * (samples.size() - 1)
                );
        }

        double value = samples[index];
        int y = midY - static_cast<int>(value * graphRect.height() * 0.45);

        if (x == 0)
            path.moveTo(graphRect.left() + x, y);
        else
            path.lineTo(graphRect.left() + x, y);
    }

    p.setPen(QPen(Qt::blue, 1));
    p.drawPath(path);

    double duration = static_cast<double>(samples.size()) / sampleRate;

    p.setPen(Qt::black);
    p.drawText(graphRect.left(), size.height() - 10, "0 s");
    p.drawText(graphRect.right() - 70, size.height() - 10,
               QString::number(duration, 'f', 2) + " s");

    return pixmap;
}

QPixmap AudioProcessor::drawSpectrum(const QVector<double> &spectrum,
                                     int sampleRate,
                                     int fftSize)
{
    QSize size(900, 260);
    QPixmap pixmap(size);
    pixmap.fill(Qt::white);

    QPainter p(&pixmap);
    p.setRenderHint(QPainter::Antialiasing);

    QRect graphRect(55, 25, size.width() - 70, size.height() - 60);

    p.setPen(Qt::black);
    p.drawRect(graphRect);
    p.drawText(10, 18, "FFT Magnitude Spectrum");

    if (spectrum.isEmpty())
        return pixmap;

    int maxFreq = std::min(10000, sampleRate / 2);
    int maxBin = static_cast<int>((static_cast<double>(maxFreq) / sampleRate) * fftSize);

    maxBin = std::min(maxBin, static_cast<int>(spectrum.size()) - 1);

    QPainterPath path;

    for (int x = 0; x < graphRect.width(); ++x) {
        int bin = static_cast<int>(
            static_cast<double>(x) / graphRect.width() * maxBin
            );

        double value = spectrum[bin];
        int y = graphRect.bottom() - static_cast<int>(value * graphRect.height());

        if (x == 0)
            path.moveTo(graphRect.left() + x, y);
        else
            path.lineTo(graphRect.left() + x, y);
    }

    p.setPen(QPen(Qt::red, 1));
    p.drawPath(path);

    p.setPen(Qt::black);
    p.drawText(graphRect.left(), size.height() - 10, "0 Hz");
    p.drawText(graphRect.right() - 90, size.height() - 10,
               QString::number(maxFreq) + " Hz");

    return pixmap;
}

QPixmap AudioProcessor::drawSpectrogram(const QVector<double> &samples,
                                        int sampleRate)
{
    QSize size(900, 260);
    QPixmap pixmap(size);
    pixmap.fill(Qt::white);

    QPainter p(&pixmap);
    p.setRenderHint(QPainter::Antialiasing);

    QRect graphRect(55, 25, size.width() - 70, size.height() - 60);

    p.setPen(Qt::black);
    p.drawRect(graphRect);
    p.drawText(10, 18, "Spectrogram");

    int frameSize = 1024;
    int hopSize = 512;

    if (samples.size() < frameSize) {
        p.drawText(graphRect, Qt::AlignCenter,
                   "Audio is too short for spectrogram.");
        return pixmap;
    }



    int sampleCount = static_cast<int>(samples.size());
    int frameCount = 1 + (sampleCount - frameSize) / hopSize;
    int maxFreq = std::min(10000, sampleRate / 2);
    int maxBin = static_cast<int>((static_cast<double>(maxFreq) / sampleRate) * frameSize);
    maxBin = std::min(maxBin, frameSize / 2 - 1);

    QVector<QVector<double>> spec(frameCount, QVector<double>(maxBin + 1));

    double globalMax = 1e-12;
    const double pi = std::acos(-1.0);

    for (int frame = 0; frame < frameCount; ++frame) {
        std::vector<std::complex<double>> buffer(frameSize);
        int start = frame * hopSize;

        for (int i = 0; i < frameSize; ++i) {
            double hann = 0.5 * (1.0 - std::cos(2.0 * pi * i / (frameSize - 1)));
            buffer[i] = std::complex<double>(samples[start + i] * hann, 0.0);
        }

        fft(buffer);

        for (int bin = 0; bin <= maxBin; ++bin) {
            double mag = std::abs(buffer[bin]);
            spec[frame][bin] = mag;

            if (mag > globalMax)
                globalMax = mag;
        }
    }

    QImage image(graphRect.width(), graphRect.height(), QImage::Format_RGB32);
    image.fill(Qt::black);

    for (int x = 0; x < graphRect.width(); ++x) {
        int frame = static_cast<int>(
            static_cast<double>(x) / graphRect.width() * (frameCount - 1)
            );

        for (int y = 0; y < graphRect.height(); ++y) {
            int bin = static_cast<int>(
                static_cast<double>(graphRect.height() - 1 - y) /
                graphRect.height() * maxBin
                );

            double value = spec[frame][bin] / globalMax;

            value = std::log10(1.0 + 50.0 * value) / std::log10(51.0);

            int shade = qBound(0, static_cast<int>(value * 255.0), 255);

            image.setPixel(x, y, QColor(shade, shade, shade).rgb());
        }
    }

    p.drawImage(graphRect.topLeft(), image);

    p.setPen(Qt::black);
    p.drawText(graphRect.left(), size.height() - 10, "Time");
    p.drawText(8, graphRect.top() + 15, QString::number(maxFreq) + " Hz");
    p.drawText(15, graphRect.bottom(), "0 Hz");

    return pixmap;
}
