#include "NeuralNetwork.hpp"
#include "NeuralNetwork_impl.hpp"
#include "NeuronActivation.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

int main() {
      // revisar las conecciones
      NetworkDimentions dims{1, {1}, 1};
      NeuralNetwork net{dims, new NeuronActivations::relu{},
                        new NeuronActivations::regression{}};
      // funcion de perdida
      net.fit({0, 9, 23, 27, 35, 38}, {32, 48, 73, 80, 95, 100}, 500000, 3);
      NeuralNetwork_impl::NetworkData input(1);
      std::cout << "\nPredicciones"
                << "\n";
      while (std::cin >> input[0]) {
            std::cout << net.predict(input)[0] << std::endl;
      }
}
