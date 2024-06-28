#include "include/Data.hpp"
#include "include/LossFuctions.hpp"
#include "include/NetworkAlgoritms.hpp"
#include "include/NeuralNetwork.hpp"
#include "include/Neuron.hpp"
#include "include/NeuronActivation.hpp"
#include "include/OptimizationAlgorithms.hpp"
#include <iostream>

int main() {
      NeuralNetwork network{
          {{Neuron::TYPE::INPUT, NeuronActivations::TYPE::SIGMOID,
            OptimizationAlgorithms::TYPE::ADAMS, 2},
           {Neuron::TYPE::WIDE, NeuronActivations::TYPE::SIGMOID,
            OptimizationAlgorithms::TYPE::ADAMS, 2},
           {Neuron::TYPE::OUTPUT, NeuronActivations::TYPE::SIGMOID,
            OptimizationAlgorithms::TYPE::ADAMS, 2}},
          LossFuctions::TYPE::MSE,
          0.1};
      // rpoblemas cuando es mas de una neurona
      network.fit({{{0, 0}, {1, 0}},
                   {{0, 1}, {0, 1}},
                   {{1, 0}, {0, 1}},
                   {{1, 1}, {1, 0}}},
                  5000, 1, 10e-5);
      // network.alphaAlgorithms.upDecayLearningRate({0.0001, 0.00001, 10});*/
      /*network.fit(
          {{
              {{-40}, {-40}}, {{-30}, {-22}}, {{-20}, {-4}}, {{-10}, {14}},
              {{0}, {32}},    {{5}, {41}},    {{10}, {50}},  {{15}, {59}},
              {{20}, {68}},   {{25}, {77}},   {{30}, {86}},  {{35}, {95}},
              {{40}, {104}},  {{45}, {113}},  {{50}, {122}}, {{55}, {131}},
              {{60}, {140}},  {{65}, {149}},  {{70}, {158}}, {{75}, {167}},
              {{80}, {176}},  {{85}, {185}},  {{90}, {194}}, {{95}, {203}},
              {{100}, {212}},
          }},
          200, 1, 10e-5);*/

      int nInputNeurons = 2;
      InputNetworkData input(nInputNeurons);
      std::cout << "\nPredicciones"
                << "\n";

      while (std::cin >> input[0] >> input[1]) {
            std::cout << network.predict(input)[0] << ">1" << std::endl;
            std::cout << network.predict(input)[1] << ">0" << std::endl;
      }
}
