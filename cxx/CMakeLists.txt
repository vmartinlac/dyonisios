
cmake_minimum_required(VERSION 3.0)
project(dyonisos)

set(CMAKE_CXX_STANDARD 14)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(dyonisos main.cpp)

target_link_libraries(dyonisos PUBLIC ${TORCH_LIBRARIES} opencv_core opencv_imgcodecs)
target_compile_definitions(dyonisos PUBLIC _GLIBCXX_USE_CXX11_ABI=1)

