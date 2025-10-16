#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class FileProcessor;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_browseButton_clicked();
    void on_startButton_clicked();
    void updateStatusMessage(const QString &message);
    void updateProgressBar(int value);
    void processingFinished();
    void onRunTypeChanged();
    void triggerProcessing();


private:
    Ui::MainWindow *ui;
    FileProcessor *processor;
    QThread *processingThread;
    QTimer *scanTimer;
};
#endif 
