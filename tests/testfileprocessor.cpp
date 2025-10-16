#include <QTest>
#include <QSignalSpy>
#include <QTemporaryDir>
#include <QTemporaryFile>
#include "fileprocessor.h"

class TestFileProcessor : public QObject
{
    Q_OBJECT

private:
    QTemporaryDir tempDir;
    FileProcessor::Settings settings;


private slots:
    void initTestCase();
    void cleanupTestCase();
    void testXorLogic();
    void testFileOverwrite();
    void testFileNameModify();
    void testDeleteInputFile();
};

void TestFileProcessor::initTestCase()
{
    QVERIFY(tempDir.isValid());
    settings.outputDir = tempDir.path();
    settings.xorKey = 0xDEADBEEFCAFEBABE;
}

void TestFileProcessor::cleanupTestCase()
{
    tempDir.remove();
}

void TestFileProcessor::testXorLogic()
{
    QByteArray originalData = "Hello, this is a test string for XOR modification!";
    quint64 xorKey = 0xDEADBEEFCAFEBABE;
    char* xorKeyBytes = reinterpret_cast<char*>(&xorKey);

    QByteArray encryptedData = originalData;
    for (int i = 0; i < encryptedData.size(); ++i) {
        encryptedData[i] = encryptedData[i] ^ xorKeyBytes[i % sizeof(quint64)];
    }

    QVERIFY(originalData != encryptedData);

    QByteArray decryptedData = encryptedData;
    for (int i = 0; i < decryptedData.size(); ++i) {
        decryptedData[i] = decryptedData[i] ^ xorKeyBytes[i % sizeof(quint64)];
    }

    QCOMPARE(decryptedData, originalData);
}

void TestFileProcessor::testFileOverwrite()
{
    QTemporaryFile inputFile;
    QVERIFY(inputFile.open());
    const QString inputFilePath = inputFile.fileName();
    inputFile.write("test data");
    inputFile.close();

    QFile existingOutputFile(settings.outputDir + "/" + QFileInfo(inputFilePath).fileName());
    QVERIFY(existingOutputFile.open(QIODevice::WriteOnly));
    const QString initialContent = "initial content";
    existingOutputFile.write(initialContent);
    existingOutputFile.close();


    FileProcessor processor;
    settings.fileMask = QFileInfo(inputFilePath).fileName();
    settings.overwrite = true;
    settings.deleteInput = false;
    processor.setSettings(settings);

    QSignalSpy spy(&processor, &FileProcessor::finished);
    processor.processFiles();
    QVERIFY(spy.wait(5000));

    QVERIFY(QFile::exists(existingOutputFile.fileName()));
    QVERIFY(existingOutputFile.open(QIODevice::ReadOnly));
    QByteArray finalContent = existingOutputFile.readAll();
    QVERIFY(finalContent != initialContent);
    QVERIFY(finalContent.size() == QString("test data").size());
}


void TestFileProcessor::testFileNameModify()
{
    QTemporaryFile inputFile;
    QVERIFY(inputFile.open());
    const QString inputFilePath = inputFile.fileName();
    inputFile.write("test data");
    inputFile.close();

    QString existingOutputFileName = settings.outputDir + "/" + QFileInfo(inputFilePath).fileName();
    QFile existingOutputFile(existingOutputFileName);
    QVERIFY(existingOutputFile.open(QIODevice::WriteOnly));
    existingOutputFile.write("initial content");
    existingOutputFile.close();

    FileProcessor processor;
    settings.fileMask = QFileInfo(inputFilePath).fileName();
    settings.overwrite = false;
    settings.deleteInput = false;
    processor.setSettings(settings);

    QSignalSpy spy(&processor, &FileProcessor::finished);
    processor.processFiles();
    QVERIFY(spy.wait(5000));

    QFileInfo fi(inputFilePath);
    QString newFileName = settings.outputDir + "/" + fi.baseName() + "_1." + fi.suffix();
    QVERIFY(QFile::exists(newFileName));
}

void TestFileProcessor::testDeleteInputFile()
{
    QTemporaryFile *inputFile = new QTemporaryFile();
    QVERIFY(inputFile->open());
    const QString inputFilePath = inputFile->fileName();
    inputFile->write("test data");
    inputFile->setAutoRemove(false);
    inputFile->close();

    FileProcessor processor;
    settings.fileMask = QFileInfo(inputFilePath).fileName();
    settings.overwrite = true;
    settings.deleteInput = true;
    processor.setSettings(settings);

    QSignalSpy spy(&processor, &FileProcessor::finished);
    processor.processFiles();
    QVERIFY(spy.wait(5000));

    QVERIFY(!QFile::exists(inputFilePath));
}



QTEST_MAIN(TestFileProcessor)
#include "testfileprocessor.moc"
