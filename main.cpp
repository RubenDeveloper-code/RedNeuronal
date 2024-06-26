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
            Algorithms::TYPE::ADAMS, 1},
           {Neuron::TYPE::OUTPUT, NeuronActivations::TYPE::REGRESSION,
            Algorithms::TYPE::ADAMS, 1}},
          LossFuctions::TYPE::MSE};
      /* network.fit({{{0, 0}, {0}}, {{0, 1}, {1}}, {{1, 0}, {1}}, {{1, 1},
         {1}}}, 500, 4);*/
      // revisar la function de perdida
      // SOLO NO FUNCIONA CUANDO SE USAN TODOS LOS DATOS DEL CONJUNTO
      network.fit(
          {{
              {{-40}, {-40}}, {{-30}, {-22}}, {{-20}, {-4}}, {{-10}, {14}},
              {{0}, {32}},    {{5}, {41}},    {{10}, {50}},  {{15}, {59}},
              {{20}, {68}},   {{25}, {77}},   {{30}, {86}},  {{35}, {95}},
              {{40}, {104}},  {{45}, {113}},  {{50}, {122}}, {{55}, {131}},
              {{60}, {140}},  {{65}, {149}},  {{70}, {158}}, {{75}, {167}},
              {{80}, {176}},  {{85}, {185}},  {{90}, {194}}, {{95}, {203}},
              {{100}, {212}},
          }},
          200, 1);

      int nInputNeurons = 1;
      InputNetworkData input(nInputNeurons);
      std::cout << "\nPredicciones"
                << "\n";

      while (std::cin >> input[0]) {
            std::cout << network.predict(input)[0] << std::endl;
      }
}
