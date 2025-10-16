#ifndef FILEPROCESSOR_H
#define FILEPROCESSOR_H

#include <QAtomicInt>
#include <QObject>
#include <QString>

class FileProcessor : public QObject {
  Q_OBJECT
public:
  struct Settings {
    QString fileMask;
    QString outputDir;
    bool deleteInput;
    bool overwrite;
    quint64 xorKey;
  };

  explicit FileProcessor(QObject *parent = nullptr);
  void setSettings(const Settings &settings);
  void stopProcessing();

public slots:
  void processFiles();

signals:
  void progressUpdated(int percentage);
  void statusUpdated(const QString &status);
  void finished();

private:
  Settings m_settings;
  QAtomicInt m_stopProcessing;
};

#endif
