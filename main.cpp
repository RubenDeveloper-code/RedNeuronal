#include "NeuralNetwork.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

int main() {
      // revisar las conecciones
      NetworkDimentions dims{2, {2}, 1};
      NeuralNetwork net{dims, [](double y) {
                              return (1.0 / (1.0 + std::exp(-y)));
                              /*return std::max(0.0, y);*/
                              /*return std::tanh(y);*/
                        }};
      // funcion de perdida
      net.fit({1, 1, 1, 0, 0, 1, 0, 0}, {1, 0, 1, 0}, 50000);
}
