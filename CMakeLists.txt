cmake_minimum_required(VERSION 3.16)
project(btune C CXX)

set(BLOSC_SRC_DIR "" CACHE PATH "Directory that contains the c-blosc2 sources")
if(NOT BLOSC_SRC_DIR)
    message( SEND_ERROR "Call with -DBLOSC_SRC_DIR=XXX" )
endif()

add_library(btune btune/btune.c)

include_directories(
    ${BLOSC_SRC_DIR}/blosc
    ${BLOSC_SRC_DIR}/include
)
target_link_directories(btune
    PUBLIC ${BLOSC_SRC_DIR}/build/blosc
)
target_link_libraries(btune
    blosc2
)
