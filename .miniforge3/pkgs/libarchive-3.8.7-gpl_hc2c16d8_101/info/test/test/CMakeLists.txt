cmake_minimum_required(VERSION 3.20)

project(test-libarchive LANGUAGES C)

add_executable(test-libarchive test.c)

find_path(libarchive_INCLUDE_DIR NAMES archive.h)
find_library(libarchive_LIBRARY NAMES "${LIB_NAME}")
target_link_libraries(test-libarchive PRIVATE ${libarchive_LIBRARY})
target_include_directories(test-libarchive PRIVATE ${libarchive_INCLUDE_DIR})

# Needed for header macro to resolve to the correct symbol names.
if(WIN32 AND LIB_NAME STREQUAL "archive_static")
    target_compile_definitions(test-libarchive PRIVATE LIBARCHIVE_STATIC)
endif()

# Needed dependency on macOS.
if(APPLE)
    find_package(Iconv REQUIRED)
    target_link_libraries(test-libarchive PRIVATE Iconv::Iconv)
endif()

enable_testing()
add_test(NAME test-libarchive COMMAND test-libarchive)
