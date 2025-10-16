cmake_minimum_required(VERSION 3.5)

project(TestTask LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt5 COMPONENTS Core Gui Widgets REQUIRED)

add_executable(TestTask
  src/main.cpp
  src/mainwindow.cpp
  src/fileprocessor.cpp
  include/mainwindow.h
  include/fileprocessor.h
  forms/mainwindow.ui
)

target_include_directories(TestTask PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_link_libraries(TestTask Qt5::Core Qt5::Gui Qt5::Widgets)
