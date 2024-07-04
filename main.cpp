#include "include/Data.hpp"
#include "include/LossFuctions.hpp"
#include "include/NeuralNetwork.hpp"
#include "include/Neuron.hpp"
#include "include/NeuronActivation.hpp"
#include "include/Normalizations.hpp"
#include "include/OptimizationAlgorithms.hpp"
#include "include/csvReader.hpp"
#include <iostream>
#include <memory>

void makeprediction(NeuralNetwork &net,
                    std::unique_ptr<Normalizations::Normalization> normalizer,
                    double nInputs, double nOutputs);
int main() {
      NeuralNetwork network{
          {{Neuron::TYPE::INPUT, NeuronActivations::TYPE::SIGMOID,
            OptimizationAlgorithms::TYPE::ADAMS, 12},
           {Neuron::TYPE::WIDE, NeuronActivations::TYPE::RELU,
            OptimizationAlgorithms::TYPE::ADAMS, 128},
           {Neuron::TYPE::WIDE, NeuronActivations::TYPE::RELU,
            OptimizationAlgorithms::TYPE::ADAMS, 64},
           {Neuron::TYPE::WIDE, NeuronActivations::TYPE::RELU,
            OptimizationAlgorithms::TYPE::ADAMS, 32},
           {Neuron::TYPE::OUTPUT, NeuronActivations::TYPE::REGRESSION,
            OptimizationAlgorithms::TYPE::ADAMS, 1}},
          LossFuctions::TYPE::MSE,
          0.001};

      CSVReader reader("../res/Student_performance_data _.csv");
      NetworkTrainData data = reader.toNetworkTrainData(
          {"Age", "Gender", "Ethnicity", "ParentalEducation", "StudyTimeWeekly",
           "Absences", "Tutoring", "ParentalSupport", "Extracurricular",
           "Sports", "Music", "Volunteering"},
          {"GradeClass"}, 100);
      auto normalizer =
          Normalizations::newInstance(Normalizations::TYPE::ZCORE);
      data = normalizer->normalizeData(data);
      network.fit(data, 700, 1, 10e-3);
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

      makeprediction(network, std::move(normalizer), data[0].input.size(),
                     data[0].output.size());
}

void makeprediction(NeuralNetwork &net,
                    std::unique_ptr<Normalizations::Normalization> normalizer,
                    double nInputs, double nOutputs) {
      InputNetworkData input(nInputs);
      std::cout << "\nPredicciones"
                << "\n";

      while (true) {
            for (auto i = 0; i < nInputs; i++) {
                  double temp;
                  std::cin >> temp;
                  input[i] = normalizer->individualNormalization(temp, i);
            }
            auto output = net.predict(input);
            for (auto i = 0; i < nOutputs; i++) {
                  double out =
                      normalizer->revertNormalization(output[i], i + nInputs);
                  std::cout << out << std::endl;
            }
      }
}
