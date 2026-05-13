cmake_minimum_required(VERSION 3.19)

project(PiAudioAnalyzer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt6 6.5 REQUIRED COMPONENTS Core Widgets Multimedia)

qt_standard_project_setup()

qt_add_executable(PiAudioAnalyzer
    WIN32 MACOSX_BUNDLE

    main.cpp

    mainwindow.cpp
    mainwindow.h
    mainwindow.ui

    audioprocessor.h
    audioprocessor.cpp

    audiodevicemanager.h
    audiodevicemanager.cpp

    realtimeaudiocontroller.h
    realtimeaudiocontroller.cpp

    ffthelper.h
    ffthelper.cpp

    stftprocessor.h
    stftprocessor.cpp

    realtimeplotrenderer.h
    realtimeplotrenderer.cpp

    featuremaprenderer.h
    featuremaprenderer.cpp

    melfeatureprocessor.h
    melfeatureprocessor.cpp

    melfilterbank.h
    melfilterbank.cpp

    melbank.h
    melbank.cpp

    melspec.h
    melspec.cpp

    logmel.h
    logmel.cpp

    mfccprocessor.h
    mfccprocessor.cpp

    realtimefeaturehelper.h
    realtimefeaturehelper.cpp

    surface3drenderer.h
    surface3drenderer.cpp

    stftnoisereducer.h
    stftnoisereducer.cpp

    normalizer.h
    normalizer.cpp

    noisereducer.h
    noisereducer.cpp
)

target_link_libraries(PiAudioAnalyzer
    PRIVATE
        Qt6::Core
        Qt6::Widgets
        Qt6::Multimedia
)

include(GNUInstallDirs)

install(TARGETS PiAudioAnalyzer
    BUNDLE  DESTINATION .
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

qt_generate_deploy_app_script(
    TARGET PiAudioAnalyzer
    OUTPUT_SCRIPT deploy_script
    NO_UNSUPPORTED_PLATFORM_ERROR
)

install(SCRIPT ${deploy_script})