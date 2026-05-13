#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QAudioDevice>
#include <QElapsedTimer>
#include <QImage>
#include <QList>
#include <QMainWindow>
#include <QString>
#include <QTimer>
#include <QVector>

#include "noisereducer.h"
#include "realtimeaudiocontroller.h"
#include "stftnoisereducer.h"
#include "stftprocessor.h"
#include "surface3drenderer.h"

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_btnUpload_clicked();
    void on_btnAnalyse_clicked();

    void on_btnRefreshDevices_clicked();
    void on_btnStartRealtime_clicked();
    void on_btnStopRealtime_clicked();

    void handleRealtimeSamples(const QVector<double> &samples, int sampleRate);
    void handleRealtimeStatus(const QString &statusText);

    void on_tabView_currentChanged(int index);

    void on_btnRecord_clicked();
    void on_btnSaveRecording_clicked();
    void updateRecordingTime();

    void toggleCleanRecording();
    void saveCleanRecording();

private:
    Ui::MainWindow *ui;
    QString audioFilePath;

    QList<QAudioDevice> inputDevices;

    RealtimeAudioController *realtimeController = nullptr;
    StftProcessor stftProcessor;

    QImage realtimeSpectrogramImage;
    QImage realtimeMelImage;
    QImage realtimeLogMelImage;
    QImage realtimeMfccImage;
    QImage realtime3DStft2DImage;

    QImage noiseOriginalImage;
    QImage noiseReducedImage;
    StftNoiseReducer stftNoiseReducer;

    NoiseReducer noiseReducer;

    Surface3DRenderer surface3DRenderer;

    bool isRecording = false;
    bool pendingRecordingMarker = false;

    QVector<double> recordedSamples;
    int recordedSampleRate = 0;

    QElapsedTimer recordingElapsed;
    QTimer recordingTimer;

    bool isCleanRecording = false;
    QVector<double> cleanRecordingBuffer;
    int cleanRecordedSampleRate = 0;
    QElapsedTimer cleanRecordingElapsed;

    void appendRecordingSamples(const QVector<double> &samples, int sampleRate);

    bool saveRecordingAsWav(const QString &filePath,
                            const QVector<double> &samples,
                            int sampleRate);

    void drawRecordingMarker(QImage &image);

    void refreshInputDevices();
};

#endif
