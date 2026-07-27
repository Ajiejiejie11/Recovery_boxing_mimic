#!/usr/bin/env bash

if [[ "${PKG_NAME}" == *static ]]; then
    SHARED="OFF"
else
    SHARED="ON"
fi

cmake -B "build-${PKG_NAME}/" \
    -G Ninja \
    -D BUILD_SHARED_LIBS="${SHARED}" \
    -D YAML_BUILD_SHARED_LIBS="${SHARED}"  \
    -D YAML_CPP_BUILD_TESTS=OFF \
    ${CMAKE_ARGS} 
cmake --build "build-${PKG_NAME}/" --parallel ${CPU_COUNT} --verbose
cmake --install "build-${PKG_NAME}/"
