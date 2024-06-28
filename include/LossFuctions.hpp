#ifndef __LOSSFUNCTIONS_HPP__
#define __LOSSFUNCTIONS_HPP__
#include "Data.hpp"
#include <cmath>
#include <iostream>
#include <memory>

namespace LossFuctions {
enum class TYPE { MSE, BINARYCROSSENTROPY };
struct LossFunction {
      virtual double function(std::vector<OutputNetworkData> activations,
                              std::vector<OutputNetworkData> targets) = 0;
      virtual double derivative(OutputNetworkData activation,
                                OutputNetworkData target) = 0;
      double summation(std::vector<OutputNetworkData> activations,
                       std::vector<OutputNetworkData> targets,
                       double (*ecuation)(double, double)) {
            const int SIZE_BATCH = activations.size();
            const int OUTPUT_SIZE = activations[0].size();
            double summation{};
            double loss{};
            for (auto itBatch = 0; itBatch < SIZE_BATCH; itBatch++) {
                  for (int itSingle = 0; itSingle < OUTPUT_SIZE; itSingle++) {
                        summation += ecuation(targets[itBatch][itSingle],
                                              activations[itBatch][itSingle]);
                  }
            }
            loss = summation / (SIZE_BATCH * OUTPUT_SIZE);
            return loss;
      }
      virtual ~LossFunction() = default;
};

struct MeanSquaredError : public LossFunction {
      double function(std::vector<OutputNetworkData> activations,
                      std::vector<OutputNetworkData> targets) override {
            double loss{};
            loss = summation(activations, targets,
                             [](double target, double activation) -> double {
                                   return std::pow(target - activation, 2.0);
                             });
            return loss;
      }
      double derivative(OutputNetworkData activation,
                        OutputNetworkData target) override {
            double summ{};
            const int N = activation.size();
            for (int it = 0; it < N; it++) {
                  summ += (2.0 / (N)) * (target[it] - activation[it]);

                  // summ += (2.0) * (target[it] - activation[it]);
            }
            return (summ);
      }
};

struct BinaryCrossEntropy : public LossFunction {
      double function(std::vector<OutputNetworkData> activations,
                      std::vector<OutputNetworkData> targets) override {
            double loss{};
            loss = summation(
                activations, targets, [](double target, double activation) {
                      return (target * log(activation + 10e-10) +
                              (1.0 - target) * log(1.0 - activation + 10e-10));
                });
            return -loss;
      }
      double derivative(OutputNetworkData activation,
                        OutputNetworkData target) override {
            double summ{};
            const int N = activation.size();
            for (int it = 0; it < N; it++) {
                  summ += -(target[it] / activation[it]) +
                          ((1.0 - target[it]) / (1.0 - activation[it]));
            }
            // std::cout << "loss" << summ / N << std::endl;
            return (summ / N);
      }
};

inline std::shared_ptr<LossFunction> newInstance(TYPE type) {
      switch (type) {
      case TYPE::MSE:
            return std::make_shared<MeanSquaredError>();
      case TYPE::BINARYCROSSENTROPY:
            return std::make_shared<BinaryCrossEntropy>();
      }
      return std::make_shared<MeanSquaredError>();
}
} // namespace LossFuctions
#endif
