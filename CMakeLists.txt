cmake_minimum_required(VERSION 3.16)
project(btune C CXX)

set(BLOSC_SRC_DIR "" CACHE PATH "Directory that contains the c-blosc2 sources")
if(NOT BLOSC_SRC_DIR)
    message( SEND_ERROR "Call with -DBLOSC_SRC_DIR=XXX" )
endif()

set(BLOSC_EP_SRC_DIR "" CACHE PATH "Directory that contains the entropy prober sources")
if(NOT BLOSC_EP_SRC_DIR)
    message( SEND_ERROR "Call with -DBLOSC_EP_SRC_DIR=XXX" )
endif()

set(TENSORFLOW_SRC_DIR "" CACHE PATH "Directory that contains the TensorFlow project")
if(NOT TENSORFLOW_SRC_DIR)
    message( SEND_ERROR "Call with -DTENSORFLOW_SRC_DIR=XXX" )
endif()

include_directories(
    ${BLOSC_SRC_DIR}/blosc
    ${BLOSC_SRC_DIR}/include
    ${BLOSC_EP_SRC_DIR}
    ${TENSORFLOW_SRC_DIR}
)

add_library(btune SHARED btune/btune.c btune/btune_model.cpp btune/json.c)
target_link_directories(btune
    PUBLIC ${BLOSC_SRC_DIR}/build/blosc
    PUBLIC ${BLOSC_EP_SRC_DIR}
    PUBLIC ${TENSORFLOW_SRC_DIR}/bazel-out/k8-opt/bin/tensorflow/lite
)
target_link_libraries(btune blosc2 blosc2_ep tensorflowlite)

add_executable(main main.c)
target_link_libraries(main btune)
