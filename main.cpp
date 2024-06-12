#include "include/Algorithm.hpp"
#include "include/Data.hpp"
#include "include/NeuralNetwork.hpp"
#include "include/Neuron.hpp"
#include "include/NeuronActivation.hpp"
#include <iostream>

// Correjir el sistema de perdida, mas elegante,
// Diferente manera de inicializacion de capas
// manejo dinamico de alfa
int main() {
      // netwoek design E/ {INPUT, ADAMS, SIGMOID, 5}
      NeuralNetwork network{
          {{Neuron::TYPE::INPUT, NeuronActivations::TYPE::SIGMOID,
            Algorithms::TYPE::DEFAULT, 1},
           {Neuron::TYPE::WIDE, NeuronActivations::TYPE::REGRESSION,
            Algorithms::TYPE::DEFAULT, 1},
           {Neuron::TYPE::OUTPUT, NeuronActivations::TYPE::REGRESSION,
            Algorithms::TYPE::DEFAULT, 1}}};
      network.fit(
          {
              {{0}, {32}},
              {{9}, {48}},
              {{23}, {73}},
              {{27}, {80}},
              {{35}, {95}},
              {{38}, {100}},
          },
          500000, 6);

      int nInputNeurons = 1;
      InputNetworkData input(nInputNeurons);
      std::cout << "\nPredicciones"
                << "\n";

      while (std::cin >> input[0]) {
            std::cout << network.predict(input)[0] << std::endl;
      }
}
