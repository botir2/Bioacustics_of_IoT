#ifndef REALTIMEAUDIOCONTROLLER_H
#define REALTIMEAUDIOCONTROLLER_H

#include <QObject>
#include <QAudioDevice>
#include <QAudioFormat>
#include <QAudioSource>
#include <QIODevice>
#include <QVector>
#include <QString>

class RealtimeAudioController : public QObject
{
    Q_OBJECT

public:
    explicit RealtimeAudioController(QObject *parent = nullptr);
    ~RealtimeAudioController();

    bool start(const QAudioDevice &device,
               int requestedSampleRate,
               QString &errorMessage);

    void stop();

    bool isRunning() const;
    int actualSampleRate() const;

signals:
    void samplesReady(const QVector<double> &samples, int sampleRate);
    void statusChanged(const QString &statusText);

private slots:
    void handleReadyRead();

private:
    QAudioSource *m_audioSource = nullptr;
    QIODevice *m_audioInput = nullptr;
    QAudioFormat m_format;

    QVector<double> decodeAudioBytes(const QByteArray &bytes) const;
};

#endif