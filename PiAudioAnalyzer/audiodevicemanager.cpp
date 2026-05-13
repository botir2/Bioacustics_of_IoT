#include "audiodevicemanager.h"

#include <QMediaDevices>

QList<QAudioDevice> AudioDeviceManager::inputDevices()
{
    return QMediaDevices::audioInputs();
}

QString AudioDeviceManager::displayName(const QAudioDevice &device, int index)
{
    QString name = device.description();

    if (name.trimmed().isEmpty()) {
        name = "Input device";
    }

    return QString::number(index) + ": " + name;
}
