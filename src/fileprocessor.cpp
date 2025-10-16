#include "fileprocessor.h"

#include <QDebug>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>
#include <QThread>

FileProcessor::FileProcessor(QObject *parent)
    : QObject(parent), m_stopProcessing(0) {}

void FileProcessor::setSettings(const Settings &settings) {
  m_settings = settings;
}

void FileProcessor::stopProcessing() { m_stopProcessing.store(1); }

void FileProcessor::processFiles() {
  emit statusUpdated("Searching for files");
  QDirIterator it(QDir::currentPath(), QStringList() << m_settings.fileMask,
                  QDir::Files, QDirIterator::Subdirectories);
  QVector<QString> filesToProcess;
  while (it.hasNext()) {
    filesToProcess.append(it.next());
  }

  if (filesToProcess.isEmpty()) {
    emit statusUpdated("No files found.");
    emit finished();
    return;
  }

  emit statusUpdated(QString("Found %1 files. Starting processing")
                         .arg(filesToProcess.size()));
  qint64 totalBytesProcessed = 0;
  qint64 totalBytes = 0;
  foreach (const QString &filePath, filesToProcess) {
    totalBytes += QFileInfo(filePath).size();
  }

  int fileCounter = 0;
  for (const QString &filePath : filesToProcess) {
    if (m_stopProcessing.load()) {
      emit statusUpdated("Stopped by user.");
      emit finished();
      return;
    }

    fileCounter++;
    emit statusUpdated(QString("Processing file %1 of %2: %3")
                           .arg(fileCounter)
                           .arg(filesToProcess.size())
                           .arg(QFileInfo(filePath).fileName()));

    QFile inputFile(filePath);
    if (!inputFile.open(QIODevice::ReadOnly)) {
      qWarning() << "Could not open input file:" << filePath;
      continue;
    }

    QFileInfo fileInfo(filePath);
    QString outputFileName = fileInfo.fileName();
    QString outputFilePath =
        m_settings.outputDir + QDir::separator() + outputFileName;

    if (!m_settings.overwrite) {
      int i = 1;
      while (QFile::exists(outputFilePath)) {
        outputFilePath = m_settings.outputDir + QDir::separator() +
                         fileInfo.baseName() + "_" + QString::number(i) + "." +
                         fileInfo.suffix();
        i++;
      }
    }

    QFile outputFile(outputFilePath);
    if (!outputFile.open(QIODevice::WriteOnly)) {
      qWarning() << "Could not open output file:" << outputFilePath;
      inputFile.close();
      continue;
    }

    const qint64 bufferSize = 1024 * 1024;
    char *buffer = new char[bufferSize];
    char *xorKeyBytes = reinterpret_cast<char *>(&m_settings.xorKey);

    qint64 bytesRead;
    while ((bytesRead = inputFile.read(buffer, bufferSize)) > 0) {
      if (m_stopProcessing.load()) {
        break;
      }
      for (qint64 i = 0; i < bytesRead; ++i) {
        buffer[i] ^= xorKeyBytes[i % sizeof(quint64)];
      }
      outputFile.write(buffer, bytesRead);

      totalBytesProcessed += bytesRead;
      emit progressUpdated(int((totalBytesProcessed * 100) / totalBytes));
    }

    delete[] buffer;
    inputFile.close();
    outputFile.close();

    if (m_stopProcessing.load()) {
      QFile::remove(outputFilePath);
      emit statusUpdated("Stopped by user.");
      emit finished();
      return;
    }

    if (m_settings.deleteInput) {
      inputFile.remove();
    }
  }

  if (!m_stopProcessing.load()) {
    emit statusUpdated("Finished successfully.");
  }
  emit finished();
}
