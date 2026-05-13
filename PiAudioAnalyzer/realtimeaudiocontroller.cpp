#include "realtimeaudiocontroller.h"

#include <cstring>

RealtimeAudioController::RealtimeAudioController(QObject *parent)
    : QObject(parent)
{
}

RealtimeAudioController::~RealtimeAudioController()
{
    if (m_audioInput) {
        disconnect(m_audioInput, nullptr, this, nullptr);
        m_audioInput = nullptr;
    }

    if (m_audioSource) {
        m_audioSource->stop();
        delete m_audioSource;
        m_audioSource = nullptr;
    }
}

bool RealtimeAudioController::start(const QAudioDevice &device,
                                    int requestedSampleRate,
                                    QString &errorMessage)
{
    stop();

    QAudioFormat format;
    format.setSampleRate(requestedSampleRate);
    format.setChannelCount(1);
    format.setSampleFormat(QAudioFormat::Int16);

    if (!device.isFormatSupported(format)) {
        format = device.preferredFormat();

        if (format.sampleFormat() != QAudioFormat::Int16 &&
            format.sampleFormat() != QAudioFormat::Float) {
            errorMessage =
                "The selected microphone format is not supported by this version.\n"
                "Supported formats: Int16 or Float.";
            return false;
        }
    }

    m_format = format;

    m_audioSource = new QAudioSource(device, m_format);
    m_audioSource->setBufferSize(8192);

    m_audioInput = m_audioSource->start();

    if (!m_audioInput) {
        errorMessage = "Could not start audio input.";
        stop();
        return false;
    }

    connect(m_audioInput, &QIODevice::readyRead,
            this, &RealtimeAudioController::handleReadyRead);

    emit statusChanged(
        "Running: " +
        device.description() +
        " | " +
        QString::number(m_format.sampleRate()) + " Hz"
    );

    return true;
}

void RealtimeAudioController::stop()
{
    if (m_audioInput) {
        disconnect(m_audioInput, nullptr, this, nullptr);
        m_audioInput = nullptr;
    }

    if (m_audioSource) {
        m_audioSource->stop();
        delete m_audioSource;
        m_audioSource = nullptr;
    }

    emit statusChanged("Stopped");
}

bool RealtimeAudioController::isRunning() const
{
    return m_audioSource != nullptr;
}

int RealtimeAudioController::actualSampleRate() const
{
    return m_format.sampleRate();
}

void RealtimeAudioController::handleReadyRead()
{
    if (!m_audioInput) {
        return;
    }

    QByteArray bytes = m_audioInput->readAll();

    QVector<double> samples = decodeAudioBytes(bytes);

    if (!samples.isEmpty()) {
        emit samplesReady(samples, m_format.sampleRate());
    }
}

QVector<double> RealtimeAudioController::decodeAudioBytes(const QByteArray &bytes) const
{
    QVector<double> samples;

    int channels = m_format.channelCount();

    if (channels <= 0) {
        return samples;
    }

    if (m_format.sampleFormat() == QAudioFormat::Int16) {
        int bytesPerSample = 2;
        int frameSize = channels * bytesPerSample;
        int frameCount = bytes.size() / frameSize;

        samples.reserve(frameCount);

        const char *data = bytes.constData();

        for (int i = 0; i < frameCount; ++i) {
            double sum = 0.0;

            for (int ch = 0; ch < channels; ++ch) {
                int pos = i * frameSize + ch * bytesPerSample;

                qint16 value = 0;
                std::memcpy(&value, data + pos, sizeof(qint16));

                sum += static_cast<double>(value) / 32768.0;
            }

            samples.append(sum / channels);
        }
    }
    else if (m_format.sampleFormat() == QAudioFormat::Float) {
        int bytesPerSample = 4;
        int frameSize = channels * bytesPerSample;
        int frameCount = bytes.size() / frameSize;

        samples.reserve(frameCount);

        const char *data = bytes.constData();

        for (int i = 0; i < frameCount; ++i) {
            double sum = 0.0;

            for (int ch = 0; ch < channels; ++ch) {
                int pos = i * frameSize + ch * bytesPerSample;

                float value = 0.0f;
                std::memcpy(&value, data + pos, sizeof(float));

                sum += static_cast<double>(value);
            }

            samples.append(sum / channels);
        }
    }

    return samples;
}