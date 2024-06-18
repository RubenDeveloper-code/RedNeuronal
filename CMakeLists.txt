cmake_minimum_required(VERSION 3.10)

project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(include)

add_executable(NeuralNetwork src/NeuralNetwork.cpp src/NeuralNetworkFit.cpp src/NeuralNetworkImpl.cpp src/Neuron.cpp main.cpp)
