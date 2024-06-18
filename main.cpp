#include "include/Algorithm.hpp"
#include "include/Data.hpp"
#include "include/LossFuctions.hpp"
#include "include/NeuralNetwork.hpp"
#include "include/Neuron.hpp"
#include "include/NeuronActivation.hpp"
#include <iostream>

// TODO: Tunear lo de perdida
// TODO: Implementar el puto Adams

int main() {
      NeuralNetwork network{
          {{Neuron::TYPE::INPUT, NeuronActivations::TYPE::SIGMOID,
            Algorithms::TYPE::SGD, 1},
           {Neuron::TYPE::WIDE, NeuronActivations::TYPE::REGRESSION,
            Algorithms::TYPE::SGD, 2},
           {Neuron::TYPE::OUTPUT, NeuronActivations::TYPE::REGRESSION,
            Algorithms::TYPE::SGD, 1}},
          LossFuctions::TYPE::MSE};
      /*network.fit({{{0, 0}, {1}}, {{0, 1}, {0}}, {{1, 0}, {0}}, {{1, 1},
         {1}}}, 50000, 4);*/
      network.fit(
          {
              {{0}, {32}},
              {{9}, {48}},
              {{23}, {73}},
              {{27}, {80}},
              {{35}, {95}},
              {{38}, {100}},
          },
          50000, 6);

      int nInputNeurons = 1;
      InputNetworkData input(nInputNeurons);
      std::cout << "\nPredicciones"
                << "\n";

      while (std::cin >> input[0]) {
            std::cout << network.predict(input)[0] << std::endl;
      }
}
