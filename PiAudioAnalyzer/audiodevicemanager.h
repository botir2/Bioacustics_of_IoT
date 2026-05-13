#ifndef AUDIODEVICEMANAGER_H
#define AUDIODEVICEMANAGER_H

#include <QAudioDevice>
#include <QList>
#include <QString>

class AudioDeviceManager
{
public:
    static QList<QAudioDevice> inputDevices();
    static QString displayName(const QAudioDevice &device, int index);
};

#endif
