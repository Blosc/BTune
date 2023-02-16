cmake_minimum_required(VERSION 3.16)
project(btune C CXX)

set(BLOSC_SRC_DIR "" CACHE PATH "Directory that contains the c-blosc2 sources")
if(NOT BLOSC_SRC_DIR)
    message( SEND_ERROR "Call with -DBLOSC_SRC_DIR=XXX" )
endif()

include_directories(
    ${BLOSC_SRC_DIR}/blosc
    ${BLOSC_SRC_DIR}/include
)

add_library(btune SHARED btune/btune.c)
target_link_directories(btune PUBLIC ${BLOSC_SRC_DIR}/build/blosc)
target_link_libraries(btune blosc2)

add_executable(main main.c)
target_link_libraries(main btune)
