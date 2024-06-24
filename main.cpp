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
            Algorithms::TYPE::ADAMS, 1},
           {Neuron::TYPE::WIDE, NeuronActivations::TYPE::REGRESSION,
            Algorithms::TYPE::SGD, 1},
           {Neuron::TYPE::OUTPUT, NeuronActivations::TYPE::REGRESSION,
            Algorithms::TYPE::SGD, 1}},
          LossFuctions::TYPE::MSE};
      /* network.fit({{{0, 0}, {0}}, {{0, 1}, {1}}, {{1, 0}, {1}}, {{1, 1},
         {1}}}, 500, 4);*/
      // revisar la function de perdida
      // SOLO NO FUNCIONA CUANDO SE USAN TODOS LOS DATOS DEL CONJUNTO
      network.fit(
          {
              {{0}, {32}},
              {{9}, {48}},
              {{23}, {73}},
              {{27}, {80}},
              {{35}, {95}},
              {{38}, {100}},
          },
          500000, 5);

      int nInputNeurons = 1;
      InputNetworkData input(nInputNeurons);
      std::cout << "\nPredicciones"
                << "\n";

      while (std::cin >> input[0]) {
            std::cout << network.predict(input)[0] << std::endl;
      }
}
