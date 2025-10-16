#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "fileprocessor.h"

#include <QFileDialog>
#include <QMessageBox>
#include <QThread>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow), processor(nullptr),
      processingThread(nullptr), scanTimer(new QTimer(this)) {
  ui->setupUi(this);
  connect(ui->runTypeComboBox, SIGNAL(currentIndexChanged(int)), this,
          SLOT(onRunTypeChanged()));
  connect(scanTimer, &QTimer::timeout, this, &MainWindow::triggerProcessing);
}

MainWindow::~MainWindow() {
  if (processingThread && processingThread->isRunning()) {
    processor->stopProcessing();
    processingThread->quit();
    processingThread->wait();
  }
  delete ui;
}

void MainWindow::on_browseButton_clicked() {
  QString dir = QFileDialog::getExistingDirectory(
      this, tr("Select Output Directory"), "",
      QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
  if (!dir.isEmpty()) {
    ui->outputDirLineEdit->setText(dir);
  }
}

void MainWindow::on_startButton_clicked() {
  if (processor) {
    scanTimer->stop();
    processor->stopProcessing();
    ui->startButton->setText("Start Processing");
    ui->settingsGroupBox->setEnabled(true);
    return;
  }

  QByteArray xorKey =
      QByteArray::fromHex(ui->xorKeyLineEdit->text().toLatin1());
  if (xorKey.size() != 8) {
    QMessageBox::warning(this, "Invalid Key", "XOR key must be 8 bytes");
    return;
  }

  if (ui->outputDirLineEdit->text().isEmpty()) {
    QMessageBox::warning(this, "Invalid Settings",
                         "Output directory must be set");
    return;
  }

  ui->settingsGroupBox->setEnabled(false);
  ui->startButton->setText("Stop Processing");

  if (ui->runTypeComboBox->currentText() == "Timer") {
    scanTimer->start(ui->scanIntervalSpinBox->value());
    triggerProcessing();
  } else {
    triggerProcessing();
  }
}

void MainWindow::triggerProcessing() {
  if (processingThread && processingThread->isRunning()) {
    return;
  }

  processingThread = new QThread;
  processor = new FileProcessor;

  FileProcessor::Settings settings;
  settings.fileMask = ui->fileMaskLineEdit->text();
  settings.outputDir = ui->outputDirLineEdit->text();
  settings.deleteInput = ui->deleteInputCheckBox->isChecked();
  settings.overwrite = ui->conflictActionComboBox->currentText() == "Overwrite";
  settings.xorKey =
      quint64(stoull(ui->xorKeyLineEdit->text().toStdString(), nullptr, 16));

  processor->setSettings(settings);
  processor->moveToThread(processingThread);

  connect(processingThread, &QThread::started, processor,
          &FileProcessor::processFiles);
  connect(processor, &FileProcessor::statusUpdated, this,
          &MainWindow::updateStatusMessage);
  connect(processor, &FileProcessor::progressUpdated, this,
          &MainWindow::updateProgressBar);
  connect(processor, &FileProcessor::finished, this,
          &MainWindow::processingFinished);
  connect(processor, &FileProcessor::finished, processingThread,
          &QThread::quit);
  connect(processor, &FileProcessor::finished, processor,
          &FileProcessor::deleteLater);
  connect(processingThread, &QThread::finished, processingThread,
          &QThread::deleteLater);

  processingThread->start();
}

void MainWindow::updateStatusMessage(const QString &message) {
  ui->statusLabel->setText("Status: " + message);
}

void MainWindow::updateProgressBar(int value) {
  ui->progressBar->setValue(value);
}

void MainWindow::processingFinished() {
  updateStatusMessage("Idle");
  updateProgressBar(0);
  processor = nullptr;
  processingThread = nullptr;

  if (ui->runTypeComboBox->currentText() != "Timer") {
    ui->startButton->setText("Start Processing");
    ui->settingsGroupBox->setEnabled(true);
  }
}

void MainWindow::onRunTypeChanged() {
  bool isTimer = ui->runTypeComboBox->currentText() == "Timer";
  ui->scanIntervalSpinBox->setEnabled(isTimer);
  ui->scanIntervalLabel->setEnabled(isTimer);
}
