#include "include/Data.hpp"
#include "include/LossFuctions.hpp"
#include "include/NetworkAlgoritms.hpp"
#include "include/NeuralNetwork.hpp"
#include "include/Neuron.hpp"
#include "include/NeuronActivation.hpp"
#include "include/OptimizationAlgorithms.hpp"
#include "include/csvReader.hpp"
#include <iostream>

int main() {
      NeuralNetwork network{
          {{Neuron::TYPE::INPUT, NeuronActivations::TYPE::SIGMOID,
            OptimizationAlgorithms::TYPE::ADAMS, 13},
           {Neuron::TYPE::WIDE, NeuronActivations::TYPE::REGRESSION,
            OptimizationAlgorithms::TYPE::ADAMS, 128},
           {Neuron::TYPE::WIDE, NeuronActivations::TYPE::RELU,
            OptimizationAlgorithms::TYPE::ADAMS, 128},
           {Neuron::TYPE::OUTPUT, NeuronActivations::TYPE::REGRESSION,
            OptimizationAlgorithms::TYPE::ADAMS, 1}},
          LossFuctions::TYPE::MSE,
          0.001};

      CSVReader reader("../res/Student_performance_data _.csv");
      NetworkTrainData data = reader.toNetworkTrainData(
          {"Age", "Gender", "Ethnicity", "ParentalEducation", "StudyTimeWeekly",
           "Absences", "Tutoring", "ParentalSupport", "Extracurricular",
           "Sports", "Music", "Volunteering", "GPA"},
          {"GradeClass"}, 100);
      network.fit(data, 2000, 32, 10e-2);
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

      int nInputNeurons = 13;
      InputNetworkData input(nInputNeurons);
      std::cout << "\nPredicciones"
                << "\n";

      while (std::cin >> input[0] >> input[1] >> input[2] >> input[3] >>
             input[4] >> input[5] >> input[6] >> input[7] >> input[8] >>
             input[9] >> input[10] >> input[11] >> input[12]) {
            std::cout << network.predict(input)[0] << std::endl;
      }
}
