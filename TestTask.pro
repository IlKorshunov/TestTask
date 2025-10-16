QT += core gui widgets testlib

TARGET = TestTask
TEMPLATE = app

DEFINES += QT_DEPRECATED_WARNINGS
INCLUDEPATH += $$PWD/include

SOURCES += \
src/main.cpp \
src/mainwindow.cpp \
src/fileprocessor.cpp

HEADERS += \
include/mainwindow.h \
include/fileprocessor.h

FORMS += \
forms/mainwindow.ui

# Test-specific configuration
CONFIG(debug, debug|release) {
# Add tests to a sub-project that can be built separately
SUBDIRS += tests
}

tests.sources = tests/testfileprocessor.cpp
tests.CONFIG += qtestlib
tests.depends = TestTask
