#include "mainwindow.h"
#include "ui_mainwindow.h"

#include "audiodevicemanager.h"
#include "audioprocessor.h"
#include "realtimeplotrenderer.h"
#include "featuremaprenderer.h"
#include "realtimefeaturehelper.h"
#include "normalizer.h"

#include <QColor>
#include <QDataStream>
#include <QFile>
#include <QFileDialog>
#include <QLabel>
#include <QMessageBox>
#include <QPainter>
#include <QPainterPath>
#include <QPen>
#include <QPixmap>
#include <QRect>
#include <QRectF>
#include <QSizePolicy>
#include <QSlider>

#include <QDebug>

// Normalisation tab state kept here so mainwindow.h does not need extra variables.
static QImage normalizationBeforeImage;
static QImage normalizationAfterImage;
static StftProcessor normalizationStftProcessor;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    ui->txtResult->setText("Upload a WAV file, then click Analyse.");

    // File-based output labels can scale their pixmaps normally.
    ui->lblWaveform->setScaledContents(true);
    ui->lblFFT->setScaledContents(true);
    ui->lblSpectrogram->setScaledContents(true);

    // Lock real-time plot labels so live pixmaps do not resize the GUI.
    auto lockPlotLabel = [](QLabel *label, int height) {
        label->setScaledContents(false);
        label->setMinimumWidth(0);
        label->setMinimumHeight(height);
        label->setMaximumHeight(height);
        label->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Fixed);
    };

    lockPlotLabel(ui->lblRealtimeWaveform, 130);
    lockPlotLabel(ui->lblRealtimeSpectrogram, 260);
    lockPlotLabel(ui->lblRealtimeMelSpectrogram, 360);
    lockPlotLabel(ui->lblRealtimeLogMelSpectrogram, 360);
    lockPlotLabel(ui->lblRealtimeMfccHeatmap, 300);

    lockPlotLabel(ui->lbl3DWaveformPreview, 160);
    lockPlotLabel(ui->lbl3DFftPreview, 180);
    lockPlotLabel(ui->lbl3DStft2DPreview, 230);
    lockPlotLabel(ui->lbl3DStftSurface, 610);

    // Noise Reduction tab plots.
    lockPlotLabel(ui->lblNoiseOriginalSpectrogram, 260);
    lockPlotLabel(ui->lblNoiseReducedSpectrogram, 260);

    // Normalisation tab plots.
    lockPlotLabel(ui->lblNormalizationBeforeSpectrogram, 260);
    lockPlotLabel(ui->lblNormalizationAfterSpectrogram, 260);

    // Keep long text labels from forcing the window wider.
    auto lockTextLabel = [](QLabel *label) {
        label->setMinimumWidth(0);
        label->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Fixed);
    };

    lockTextLabel(ui->lblRealtimeStatus);
    lockTextLabel(ui->lblRealtimeMetrics);
    lockTextLabel(ui->lblRealtimeMelMetrics);
    lockTextLabel(ui->lblRealtimeLogMelMetrics);
    lockTextLabel(ui->lblFeatureModeStatus);
    lockTextLabel(ui->lbl3DStftMetrics);
    lockTextLabel(ui->lblNoiseReductionMetrics);
    lockTextLabel(ui->lblNormalizationMetrics);
    lockTextLabel(ui->lblRecordTime);

    ui->btnStopRealtime->setEnabled(false);
    ui->btnSaveRecording->setEnabled(false);
    ui->lblRecordTime->setText("Recording time: 00:00");

    connect(&recordingTimer, &QTimer::timeout,
            this, &MainWindow::updateRecordingTime);

    realtimeController = new RealtimeAudioController(this);

    connect(realtimeController, &RealtimeAudioController::samplesReady,
            this, &MainWindow::handleRealtimeSamples);

    connect(realtimeController, &RealtimeAudioController::statusChanged,
            this, &MainWindow::handleRealtimeStatus);

    connect(ui->btnRecordClean, &QPushButton::clicked,
            this, &MainWindow::toggleCleanRecording);

    connect(ui->btnSaveCleanRecording, &QPushButton::clicked,
            this, &MainWindow::saveCleanRecording);

    refreshInputDevices();

    // Update the displayed numeric values beside the noise-reduction sliders.
    auto updateNoiseLabels = [this]() {
        ui->lblNoiseThresholdValue->setText(
            QString::number(ui->sliderNoiseThreshold->value() / 100.0, 'f', 2)
            );

        ui->lblNoiseStrengthValue->setText(
            QString::number(ui->sliderNoiseStrength->value() / 100.0, 'f', 2)
            );

        ui->lblNoiseSmoothingValue->setText(
            QString::number(ui->sliderNoiseSmoothing->value() / 100.0, 'f', 2)
            );
    };

    connect(ui->sliderNoiseThreshold, &QSlider::valueChanged,
            this, [updateNoiseLabels](int) { updateNoiseLabels(); });

    connect(ui->sliderNoiseStrength, &QSlider::valueChanged,
            this, [updateNoiseLabels](int) { updateNoiseLabels(); });

    connect(ui->sliderNoiseSmoothing, &QSlider::valueChanged,
            this, [updateNoiseLabels](int) { updateNoiseLabels(); });

    updateNoiseLabels();

    on_tabView_currentChanged(ui->tabView->currentIndex());
}

MainWindow::~MainWindow()
{
    if (realtimeController) {
        realtimeController->stop();
    }

    delete ui;
}

void MainWindow::on_btnUpload_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(
        this,
        "Open WAV Audio File",
        "",
        "WAV Files (*.wav)"
        );

    if (fileName.isEmpty()) {
        return;
    }

    audioFilePath = fileName;
    ui->txtFilePath->setText(fileName);

    ui->txtResult->setText(
        "Selected WAV file:\n" + fileName +
        "\n\nReady for FFT and spectrogram analysis."
        );
}

void MainWindow::on_btnAnalyse_clicked()
{
    if (audioFilePath.isEmpty()) {
        QMessageBox::warning(this, "No file", "Please upload a WAV file first.");
        return;
    }

    AnalysisResult result = AudioProcessor::analyse(audioFilePath);

    if (!result.success) {
        QMessageBox::critical(this, "Analysis error", result.message);
        return;
    }

    ui->lblWaveform->setPixmap(result.waveformPixmap);
    ui->lblFFT->setPixmap(result.fftPixmap);
    ui->lblSpectrogram->setPixmap(result.spectrogramPixmap);

    ui->txtResult->setText(result.message);
}

void MainWindow::refreshInputDevices()
{
    inputDevices = AudioDeviceManager::inputDevices();

    ui->cmbInputDevice->clear();

    for (int i = 0; i < inputDevices.size(); ++i) {
        ui->cmbInputDevice->addItem(
            AudioDeviceManager::displayName(inputDevices[i], i)
            );
    }

    if (inputDevices.isEmpty()) {
        ui->lblRealtimeStatus->setText("No input devices found.");
    } else {
        ui->lblRealtimeStatus->setText(
            QString::number(inputDevices.size()) + " input device(s) found."
            );
    }
}

void MainWindow::on_btnRefreshDevices_clicked()
{
    refreshInputDevices();
}

void MainWindow::on_btnStartRealtime_clicked()
{
    int index = ui->cmbInputDevice->currentIndex();

    if (index < 0 || index >= inputDevices.size()) {
        QMessageBox::warning(this, "No microphone", "Please select an input device first.");
        return;
    }

    int requestedSampleRate = ui->cmbSampleRate->currentText().toInt();
    int windowSize = ui->cmbWindowSize->currentText().toInt();
    int hopSize = ui->cmbHopSize->currentText().toInt();

    QString errorMessage;

    bool ok = realtimeController->start(
        inputDevices[index],
        requestedSampleRate,
        errorMessage
        );

    if (!ok) {
        QMessageBox::critical(this, "Real-time audio error", errorMessage);
        return;
    }

    int actualSampleRate = realtimeController->actualSampleRate();

    stftProcessor.configure(actualSampleRate, windowSize, hopSize);

    realtimeSpectrogramImage =
        RealtimePlotRenderer::createSpectrogramImage(
            ui->lblRealtimeSpectrogram->size()
            );

    realtimeMelImage =
        FeatureMapRenderer::createImage(
            ui->lblRealtimeMelSpectrogram->size()
            );

    realtimeLogMelImage =
        FeatureMapRenderer::createImage(
            ui->lblRealtimeLogMelSpectrogram->size()
            );

    realtimeMfccImage =
        FeatureMapRenderer::createImage(
            ui->lblRealtimeMfccHeatmap->size()
            );

    realtime3DStft2DImage =
        RealtimePlotRenderer::createSpectrogramImage(
            ui->lbl3DStft2DPreview->size()
            );

    noiseOriginalImage =
        FeatureMapRenderer::createImage(
            ui->lblNoiseOriginalSpectrogram->size()
            );

    noiseReducedImage =
        FeatureMapRenderer::createImage(
            ui->lblNoiseReducedSpectrogram->size()
            );

    normalizationBeforeImage =
        FeatureMapRenderer::createImage(
            ui->lblNormalizationBeforeSpectrogram->size()
            );

    normalizationAfterImage =
        FeatureMapRenderer::createImage(
            ui->lblNormalizationAfterSpectrogram->size()
            );

    normalizationStftProcessor.configure(actualSampleRate, windowSize, hopSize);
    normalizationStftProcessor.reset();

    surface3DRenderer.reset();
    stftNoiseReducer.reset();

    ui->btnStartRealtime->setEnabled(false);
    ui->btnStopRealtime->setEnabled(true);

    QWidget *currentTab = ui->tabView->currentWidget();

    if (currentTab != ui->tabRealtime &&
        currentTab != ui->tabRealtimeMel &&
        currentTab != ui->tabRealtimeLogMel &&
        currentTab != ui->tabRealtimeMFCC &&
        currentTab != ui->tab3DStftSurface &&
        currentTab != ui->tabNormalization &&
        currentTab != ui->tabNoiseReduction) {
        ui->tabView->setCurrentWidget(ui->tabRealtime);
    }

    on_tabView_currentChanged(ui->tabView->currentIndex());

    ui->txtResult->setText(
        "Real-time analysis started.\n\n"
        "Requested sample rate: " + QString::number(requestedSampleRate) + " Hz\n"
                                                 "Actual sample rate: " + QString::number(actualSampleRate) + " Hz\n"
                                              "Window size: " + QString::number(windowSize) + "\n"
                                        "Hop size: " + QString::number(hopSize)
        );
}

void MainWindow::on_btnStopRealtime_clicked()
{
    realtimeController->stop();

    if (isRecording) {
        isRecording = false;
        recordingTimer.stop();

        ui->btnRecord->setText("REC");
        ui->btnSaveRecording->setEnabled(!recordedSamples.isEmpty());
    }

    ui->btnStartRealtime->setEnabled(true);
    ui->btnStopRealtime->setEnabled(false);
}

void MainWindow::handleRealtimeStatus(const QString &statusText)
{
    ui->lblRealtimeStatus->setText(statusText);
}

void MainWindow::on_tabView_currentChanged(int index)
{
    QWidget *currentTab = ui->tabView->widget(index);

    if (currentTab == ui->tabRealtimeMel) {
        ui->groupFeatureControls->show();
        ui->stackFeatureParams->setCurrentWidget(ui->pageMelParams);
        ui->lblFeatureModeStatus->setText("Selected feature: Mel Spectrogram");
        return;
    }

    if (currentTab == ui->tabRealtimeLogMel) {
        ui->groupFeatureControls->show();
        ui->stackFeatureParams->setCurrentWidget(ui->pageLogMelParams);
        ui->lblFeatureModeStatus->setText("Selected feature: Log-Mel Spectrogram");
        return;
    }

    if (currentTab == ui->tabRealtimeMFCC) {
        ui->groupFeatureControls->show();
        ui->stackFeatureParams->setCurrentWidget(ui->pageMfccParams);
        ui->lblFeatureModeStatus->setText("Selected feature: MFCC");
        return;
    }

    if (currentTab == ui->tabNoiseReduction) {
        ui->groupFeatureControls->show();
        ui->stackFeatureParams->setCurrentWidget(ui->pageNoiseReduceParams);
        ui->lblFeatureModeStatus->setText("Selected feature: Noise Reduction");
        return;
    }

    if (currentTab == ui->tabNormalization) {
        ui->groupFeatureControls->hide();
        ui->lblNormalizationMetrics->setText(
            "Normalisation: select a method and click Start Real-time."
            );
        return;
    }

    if (currentTab == ui->tab3DStftSurface) {
        ui->groupFeatureControls->hide();
        ui->lbl3DStftMetrics->setText(
            "3D STFT Surface: X = Time, Y = Frequency, Z = Magnitude / dB"
            );
        return;
    }

    ui->groupFeatureControls->hide();
}

void MainWindow::handleRealtimeSamples(const QVector<double> &samples, int sampleRate)
{
    appendRecordingSamples(samples, sampleRate);

    // Clean recording path: apply the current Noise Reduction settings
    // to the incoming microphone samples and store the cleaned audio.
    if (isCleanRecording) {
        if (!ui->chkNoiseReduce->isChecked()) {
            return;
        }

        float threshold = ui->sliderNoiseThreshold->value() / 100.0f;
        float strength  = ui->sliderNoiseStrength->value() / 100.0f;
        float smoothing = ui->sliderNoiseSmoothing->value() / 100.0f;

        bool nonStationary =
            ui->cmbNoiseReduceMode->currentText() == "Non-stationary";

        QVector<float> inputFloat;
        inputFloat.reserve(samples.size());

        for (double sample : samples) {
            inputFloat.append(static_cast<float>(sample));
        }

        QVector<float> cleanFloat =
            noiseReducer.reduce(inputFloat,
                                threshold,
                                strength,
                                smoothing,
                                nonStationary);

        if (cleanRecordedSampleRate <= 0) {
            cleanRecordedSampleRate = sampleRate;
        }

        for (float sample : cleanFloat) {
            cleanRecordingBuffer.append(static_cast<double>(sample));
        }

        qint64 totalSeconds = cleanRecordingElapsed.elapsed() / 1000;
        int minutes = static_cast<int>(totalSeconds / 60);
        int seconds = static_cast<int>(totalSeconds % 60);

        ui->lblCleanRecordTime->setText(
            QString("Clean recording time: %1:%2")
                .arg(minutes, 2, 10, QChar('0'))
                .arg(seconds, 2, 10, QChar('0'))
            );
    }




    QVector<StftFrame> frames = stftProcessor.processSamples(samples);

    if (frames.isEmpty()) {
        return;
    }

    StftFrame lastFrame = frames.last();
    QWidget *currentTab = ui->tabView->currentWidget();

    int fftSize = static_cast<int>(lastFrame.magnitude.size()) * 2;

    if (currentTab == ui->tabNormalization) {
        QString method = ui->cmbNormalizationMethod->currentText();
        QVector<double> normalized = Normalizer::apply(samples, method);

        QVector<StftFrame> normalizedFrames =
            normalizationStftProcessor.processSamples(normalized);

        if (normalizedFrames.isEmpty()) {
            return;
        }

        StftFrame normalizedLastFrame = normalizedFrames.last();


        for (const StftFrame &frame : frames) {
            FeatureMapRenderer::appendColumn(
                normalizationBeforeImage,
                frame.magnitude,
                FeatureMapColourStyle::LogMelHot
                );
        }

        for (const StftFrame &frame : normalizedFrames) {
            FeatureMapRenderer::appendColumn(
                normalizationAfterImage,
                frame.magnitude,
                FeatureMapColourStyle::LogMelHot
                );
        }

        ui->lblNormalizationBeforeSpectrogram->setPixmap(
            FeatureMapRenderer::toPixmap(normalizationBeforeImage).scaled(
                ui->lblNormalizationBeforeSpectrogram->size(),
                Qt::IgnoreAspectRatio,
                Qt::SmoothTransformation
                )
            );

        ui->lblNormalizationAfterSpectrogram->setPixmap(
            FeatureMapRenderer::toPixmap(normalizationAfterImage).scaled(
                ui->lblNormalizationAfterSpectrogram->size(),
                Qt::IgnoreAspectRatio,
                Qt::SmoothTransformation
                )
            );

        ui->lblNormalizationMetrics->setText(
            "Normalisation | Method: " + method +
            " | Before peak: " + QString::number(lastFrame.peak, 'f', 4) +
            " | After peak: " + QString::number(normalizedLastFrame.peak, 'f', 4) +
            " | Before RMS: " + QString::number(lastFrame.rms, 'f', 4) +
            " | After RMS: " + QString::number(normalizedLastFrame.rms, 'f', 4)
            );

        return;
    }

    if (currentTab == ui->tabRealtime) {
        QPixmap waveformPixmap =
            RealtimePlotRenderer::drawWaveform(
                lastFrame.waveform,
                ui->lblRealtimeWaveform->size()
                );

        ui->lblRealtimeWaveform->setPixmap(
            waveformPixmap.scaled(
                ui->lblRealtimeWaveform->size(),
                Qt::IgnoreAspectRatio,
                Qt::SmoothTransformation
                )
            );

        for (const StftFrame &frame : frames) {
            RealtimePlotRenderer::appendSpectrogramColumn(
                realtimeSpectrogramImage,
                frame.magnitude
                );
        }

        if (pendingRecordingMarker) {
            drawRecordingMarker(realtimeSpectrogramImage);
            pendingRecordingMarker = false;
        }

        ui->lblRealtimeSpectrogram->setPixmap(
            RealtimePlotRenderer::toPixmap(realtimeSpectrogramImage).scaled(
                ui->lblRealtimeSpectrogram->size(),
                Qt::IgnoreAspectRatio,
                Qt::SmoothTransformation
                )
            );

        ui->lblRealtimeMetrics->setText(
            "RMS: " + QString::number(lastFrame.rms, 'f', 5) +
            " | Peak: " + QString::number(lastFrame.peak, 'f', 5) +
            " | Dominant frequency: " +
            QString::number(lastFrame.dominantFrequency, 'f', 1) + " Hz" +
            " | Sample rate: " + QString::number(sampleRate) + " Hz"
            );

        return;
    }

    if (currentTab == ui->tabRealtimeMel) {
        int melFilters = ui->cmbMelFilterCount->currentText().toInt();
        int minFreq = ui->spinMelMinFreq->value();
        int maxFreq = ui->spinMelMaxFreq->value();
        bool usePower = ui->cmbMelPowerMode->currentText() == "Power";

        RealtimeFeatureHelper::updateMelImage(
            realtimeMelImage,
            frames,
            sampleRate,
            fftSize,
            melFilters,
            minFreq,
            maxFreq,
            usePower
            );

        ui->lblRealtimeMelSpectrogram->setPixmap(
            FeatureMapRenderer::toPixmap(realtimeMelImage).scaled(
                ui->lblRealtimeMelSpectrogram->size(),
                Qt::IgnoreAspectRatio,
                Qt::SmoothTransformation
                )
            );

        ui->lblRealtimeMelMetrics->setText(
            "Mel filters: " + QString::number(melFilters) +
            " | Min freq: " + QString::number(minFreq) + " Hz" +
            " | Max freq: " + QString::number(maxFreq) + " Hz" +
            " | Mode: " + ui->cmbMelPowerMode->currentText()
            );

        return;
    }

    if (currentTab == ui->tabRealtimeLogMel) {
        int melFilters = ui->cmbLogMelFilterCount->currentText().toInt();
        int minFreq = ui->spinLogMelMinFreq->value();
        int maxFreq = ui->spinLogMelMaxFreq->value();
        double epsilon = ui->cmbLogMelEpsilon->currentText().toDouble();
        bool useDb = ui->cmbLogMelDisplayMode->currentText() == "dB";

        RealtimeFeatureHelper::updateLogMelImage(
            realtimeLogMelImage,
            frames,
            sampleRate,
            fftSize,
            melFilters,
            minFreq,
            maxFreq,
            epsilon,
            useDb
            );

        ui->lblRealtimeLogMelSpectrogram->setPixmap(
            FeatureMapRenderer::toPixmap(realtimeLogMelImage).scaled(
                ui->lblRealtimeLogMelSpectrogram->size(),
                Qt::IgnoreAspectRatio,
                Qt::SmoothTransformation
                )
            );

        ui->lblRealtimeLogMelMetrics->setText(
            "Log-Mel filters: " + QString::number(melFilters) +
            " | Min freq: " + QString::number(minFreq) + " Hz" +
            " | Max freq: " + QString::number(maxFreq) + " Hz" +
            " | Epsilon: " + ui->cmbLogMelEpsilon->currentText() +
            " | Display: " + ui->cmbLogMelDisplayMode->currentText()
            );

        return;
    }

    if (currentTab == ui->tabRealtimeMFCC) {
        int coeffCount = ui->cmbMfccCoeffCount->currentText().toInt();
        int melFilters = ui->cmbMfccMelFilterCount->currentText().toInt();
        int minFreq = ui->spinMfccMinFreq->value();
        int maxFreq = ui->spinMfccMaxFreq->value();
        bool includeC0 = ui->chkMfccUseC0->isChecked();

        QVector<double> lastMfccValues =
            RealtimeFeatureHelper::updateMfccImage(
                realtimeMfccImage,
                frames,
                sampleRate,
                fftSize,
                melFilters,
                coeffCount,
                minFreq,
                maxFreq,
                includeC0
                );

        ui->lblRealtimeMfccHeatmap->setPixmap(
            FeatureMapRenderer::toPixmap(realtimeMfccImage).scaled(
                ui->lblRealtimeMfccHeatmap->size(),
                Qt::IgnoreAspectRatio,
                Qt::SmoothTransformation
                )
            );

        QString text = "Current MFCC coefficients:\n";

        for (int i = 0; i < lastMfccValues.size(); ++i) {
            int coeffIndex = includeC0 ? i : i + 1;

            text += "C" + QString::number(coeffIndex) +
                    " = " + QString::number(lastMfccValues[i], 'f', 4) +
                    "\n";
        }

        ui->txtRealtimeMfccValues->setText(text);

        return;
    }

    if (currentTab == ui->tab3DStftSurface) {
        QPixmap waveformPixmap =
            RealtimePlotRenderer::drawWaveform(
                lastFrame.waveform,
                ui->lbl3DWaveformPreview->size()
                );

        ui->lbl3DWaveformPreview->setPixmap(
            waveformPixmap.scaled(
                ui->lbl3DWaveformPreview->size(),
                Qt::IgnoreAspectRatio,
                Qt::SmoothTransformation
                )
            );

        QImage fftImage(ui->lbl3DFftPreview->size(), QImage::Format_RGB32);
        fftImage.fill(QColor("#020617"));

        {
            QPainter painter(&fftImage);
            painter.setRenderHint(QPainter::Antialiasing);

            QRect rect(45, 20, fftImage.width() - 70, fftImage.height() - 45);

            painter.setPen(QColor("#334155"));
            painter.drawRect(rect);

            painter.setPen(QColor("#22C55E"));

            QPainterPath path;

            for (int x = 0; x < rect.width(); ++x) {
                int bin = static_cast<int>(
                    static_cast<double>(x) / rect.width() *
                    (lastFrame.magnitude.size() - 1)
                    );

                double value = qBound(0.0, lastFrame.magnitude[bin], 1.0);
                int y = rect.bottom() - static_cast<int>(value * rect.height());

                if (x == 0) {
                    path.moveTo(rect.left() + x, y);
                } else {
                    path.lineTo(rect.left() + x, y);
                }
            }

            painter.drawPath(path);

            painter.setPen(QColor("#E5E7EB"));
            painter.drawText(8, 15, "FFT Spectrum");
        }

        ui->lbl3DFftPreview->setPixmap(
            QPixmap::fromImage(fftImage).scaled(
                ui->lbl3DFftPreview->size(),
                Qt::IgnoreAspectRatio,
                Qt::SmoothTransformation
                )
            );

        for (const StftFrame &frame : frames) {
            RealtimePlotRenderer::appendSpectrogramColumn(
                realtime3DStft2DImage,
                frame.magnitude
                );
        }

        if (pendingRecordingMarker) {
            drawRecordingMarker(realtime3DStft2DImage);
            pendingRecordingMarker = false;
        }

        ui->lbl3DStft2DPreview->setPixmap(
            RealtimePlotRenderer::toPixmap(realtime3DStft2DImage).scaled(
                ui->lbl3DStft2DPreview->size(),
                Qt::IgnoreAspectRatio,
                Qt::SmoothTransformation
                )
            );

        surface3DRenderer.appendFrames(frames, 120);

        QPixmap surfacePixmap =
            surface3DRenderer.renderSurface(
                ui->lbl3DStftSurface->size(),
                sampleRate,
                8000
                );

        ui->lbl3DStftSurface->setPixmap(
            surfacePixmap.scaled(
                ui->lbl3DStftSurface->size(),
                Qt::IgnoreAspectRatio,
                Qt::SmoothTransformation
                )
            );

        ui->lbl3DStftMetrics->setText(
            "3D STFT Surface | Sample rate: " + QString::number(sampleRate) +
            " Hz | FFT size: " + QString::number(fftSize) +
            " | History: 120 frames | Max frequency: 8000 Hz"
            );

        return;
    }

    if (currentTab == ui->tabNoiseReduction) {
        double threshold = ui->sliderNoiseThreshold->value() / 100.0;
        double strength = ui->sliderNoiseStrength->value() / 100.0;
        double smoothing = ui->sliderNoiseSmoothing->value() / 100.0;

        bool enabled = ui->chkNoiseReduce->isChecked();

        bool nonStationary =
            ui->cmbNoiseReduceMode->currentText() == "Non-stationary";

        for (const StftFrame &frame : frames) {
            FeatureMapRenderer::appendColumn(
                noiseOriginalImage,
                frame.magnitude,
                FeatureMapColourStyle::LogMelHot
                );

            QVector<double> reducedMagnitude = frame.magnitude;

            if (enabled) {
                reducedMagnitude =
                    stftNoiseReducer.processMagnitude(
                        frame.magnitude,
                        threshold,
                        strength,
                        smoothing,
                        nonStationary
                        );
            }

            FeatureMapRenderer::appendColumn(
                noiseReducedImage,
                reducedMagnitude,
                FeatureMapColourStyle::LogMelHot
                );
        }

        ui->lblNoiseOriginalSpectrogram->setPixmap(
            FeatureMapRenderer::toPixmap(noiseOriginalImage).scaled(
                ui->lblNoiseOriginalSpectrogram->size(),
                Qt::IgnoreAspectRatio,
                Qt::SmoothTransformation
                )
            );

        ui->lblNoiseReducedSpectrogram->setPixmap(
            FeatureMapRenderer::toPixmap(noiseReducedImage).scaled(
                ui->lblNoiseReducedSpectrogram->size(),
                Qt::IgnoreAspectRatio,
                Qt::SmoothTransformation
                )
            );

        ui->lblNoiseReductionMetrics->setText(
            "Noise Reduction | Enabled: " +
            QString(enabled ? "Yes" : "No") +
            " | Mode: " + ui->cmbNoiseReduceMode->currentText() +
            " | Threshold: " + QString::number(threshold, 'f', 2) +
            " | Strength: " + QString::number(strength, 'f', 2) +
            " | Smoothing: " + QString::number(smoothing, 'f', 2)
            );

        return;
    }
}

void MainWindow::on_btnRecord_clicked()
{
    if (!realtimeController || !realtimeController->isRunning()) {
        QMessageBox::warning(this,
                             "Real-time is not running",
                             "Please click Start Real-time before recording.");
        return;
    }

    if (!isRecording) {
        recordedSamples.clear();
        recordedSampleRate = realtimeController->actualSampleRate();

        isRecording = true;
        pendingRecordingMarker = true;

        recordingElapsed.restart();
        recordingTimer.start(500);

        ui->btnRecord->setText("Stop Rec");
        ui->btnSaveRecording->setEnabled(true);
        ui->lblRecordTime->setText("Recording time: 00:00");

        ui->txtResult->setText(
            "Recording started.\n\n"
            "Real-time analysis is still running."
            );

        return;
    }

    isRecording = false;
    recordingTimer.stop();

    ui->btnRecord->setText("REC");
    ui->btnSaveRecording->setEnabled(!recordedSamples.isEmpty());

    ui->txtResult->setText(
        "Recording stopped.\n\n"
        "Samples recorded: " + QString::number(recordedSamples.size()) + "\n"
                                                    "Sample rate: " + QString::number(recordedSampleRate) + " Hz\n\n"
                                                "Click Save WAV to export the recording."
        );
}

void MainWindow::on_btnSaveRecording_clicked()
{
    if (recordedSamples.isEmpty() || recordedSampleRate <= 0) {
        QMessageBox::warning(this,
                             "No recording",
                             "There is no recorded audio to save.");
        return;
    }

    QString filePath = QFileDialog::getSaveFileName(
        this,
        "Save recorded audio",
        "/home/pi/sounds/recording.wav",
        "WAV Files (*.wav)"
        );

    if (filePath.isEmpty()) {
        return;
    }

    if (!filePath.endsWith(".wav", Qt::CaseInsensitive)) {
        filePath += ".wav";
    }

    QVector<double> samplesToSave = recordedSamples;
    int sampleRateToSave = recordedSampleRate;

    bool ok = saveRecordingAsWav(filePath, samplesToSave, sampleRateToSave);

    if (!ok) {
        QMessageBox::critical(this,
                              "Save error",
                              "Could not save the WAV file.");
        return;
    }

    ui->txtResult->setText(
        "Recording saved while real-time analysis continues.\n\n"
        "Saved file:\n" + filePath + "\n\n"
                     "Saved samples: " + QString::number(samplesToSave.size()) + "\n"
                                                  "Sample rate: " + QString::number(sampleRateToSave) + " Hz"
        );
}

void MainWindow::updateRecordingTime()
{
    if (!isRecording) {
        return;
    }

    qint64 totalSeconds = recordingElapsed.elapsed() / 1000;
    int minutes = static_cast<int>(totalSeconds / 60);
    int seconds = static_cast<int>(totalSeconds % 60);

    QString text =
        QString("Recording time: %1:%2")
            .arg(minutes, 2, 10, QChar('0'))
            .arg(seconds, 2, 10, QChar('0'));

    ui->lblRecordTime->setText(text);
}

void MainWindow::appendRecordingSamples(const QVector<double> &samples,
                                        int sampleRate)
{
    if (!isRecording) {
        return;
    }

    if (recordedSampleRate <= 0) {
        recordedSampleRate = sampleRate;
    }

    for (double sample : samples) {
        recordedSamples.append(sample);
    }
}

bool MainWindow::saveRecordingAsWav(const QString &filePath,
                                    const QVector<double> &samples,
                                    int sampleRate)
{
    if (samples.isEmpty() || sampleRate <= 0) {
        return false;
    }

    QFile file(filePath);

    if (!file.open(QIODevice::WriteOnly)) {
        return false;
    }

    QDataStream out(&file);
    out.setByteOrder(QDataStream::LittleEndian);

    const quint16 audioFormat = 1;
    const quint16 channels = 1;
    const quint16 bitsPerSample = 16;
    const quint16 blockAlign = channels * bitsPerSample / 8;
    const quint32 byteRate = sampleRate * blockAlign;
    const quint32 dataSize = samples.size() * blockAlign;
    const quint32 chunkSize = 36 + dataSize;

    out.writeRawData("RIFF", 4);
    out << chunkSize;
    out.writeRawData("WAVE", 4);

    out.writeRawData("fmt ", 4);
    out << quint32(16);
    out << audioFormat;
    out << channels;
    out << quint32(sampleRate);
    out << byteRate;
    out << blockAlign;
    out << bitsPerSample;

    out.writeRawData("data", 4);
    out << dataSize;

    for (double sample : samples) {
        double clipped = qBound(-1.0, sample, 1.0);
        qint16 pcm = static_cast<qint16>(clipped * 32767.0);
        out << pcm;
    }

    file.close();
    return true;
}

void MainWindow::drawRecordingMarker(QImage &image)
{
    if (image.isNull()) {
        return;
    }

    QPainter painter(&image);

    int x = image.width() - 2;

    painter.setPen(QPen(QColor("#22D3EE"), 3));
    painter.drawLine(x, 0, x, image.height());

    painter.setPen(QPen(QColor("#FFFFFF"), 1));
    painter.drawLine(x - 3, 0, x - 3, image.height());
}


void MainWindow::toggleCleanRecording()
{
    if (!realtimeController || !realtimeController->isRunning()) {
        QMessageBox::warning(this,
                             "Real-time is not running",
                             "Please click Start Real-time before clean recording.");
        return;
    }

    if (!isCleanRecording && !ui->chkNoiseReduce->isChecked()) {
        QMessageBox::warning(this,
                             "Noise Reduction is disabled",
                             "Please enable Noise Reduction before clean recording.");
        return;
    }

    isCleanRecording = !isCleanRecording;

    if (isCleanRecording) {
        cleanRecordingBuffer.clear();
        cleanRecordedSampleRate = realtimeController->actualSampleRate();
        cleanRecordingElapsed.restart();

        ui->btnRecordClean->setText("Stop Clean");
        ui->lblCleanRecordTime->setText("Clean recording time: 00:00");
        ui->txtResult->setText(
            "Clean noise-reduced recording started.\n\n"
            "The current Noise Reduction settings will be applied to the saved WAV."
            );
    } else {
        ui->btnRecordClean->setText("REC Clean");
        ui->lblCleanRecordTime->setText("Clean recording: stopped");
        ui->txtResult->setText(
            "Clean noise-reduced recording stopped.\n\n"
            "Click Save Clean WAV to export the cleaned recording."
            );
    }
}

void MainWindow::saveCleanRecording()
{
    if (isCleanRecording) {
        isCleanRecording = false;
        ui->btnRecordClean->setText("REC Clean");
        ui->lblCleanRecordTime->setText("Clean recording: stopped");
    }

    if (cleanRecordingBuffer.isEmpty() || cleanRecordedSampleRate <= 0) {
        QMessageBox::warning(this,
                             "No clean recording",
                             "There is no noise-reduced audio to save.");
        return;
    }

    QString filePath = QFileDialog::getSaveFileName(
        this,
        "Save clean noise-reduced audio",
        "/home/pi/sounds/clean_noise_reduced.wav",
        "WAV Files (*.wav)"
        );

    if (filePath.isEmpty()) {
        return;
    }

    if (!filePath.endsWith(".wav", Qt::CaseInsensitive)) {
        filePath += ".wav";
    }

    bool ok = saveRecordingAsWav(filePath,
                                 cleanRecordingBuffer,
                                 cleanRecordedSampleRate);

    if (!ok) {
        QMessageBox::critical(this,
                              "Save error",
                              "Could not save the clean WAV file.");
        return;
    }

    ui->txtResult->setText(
        "Clean noise-reduced recording saved.\n\n"
        "Saved file:\n" + filePath + "\n\n"
                     "Saved samples: " + QString::number(cleanRecordingBuffer.size()) + "\n"
                                                         "Sample rate: " + QString::number(cleanRecordedSampleRate) + " Hz"
        );
}
