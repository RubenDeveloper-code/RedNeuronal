#ifndef __LOSSFUNCTIONS_HPP__
#define __LOSSFUNCTIONS_HPP__
#include "../data/Data.hpp"
#include <cmath>
#include <memory>

namespace LossFuctions {
enum class TYPE { MEAN_SQUARED_ERROR, BINARY_CROSS_ENTROPY };
struct LossFunction {
      virtual double function(OutputNetworkData activations,
                              OutputNetworkData targets) = 0;
      virtual double derivative(OutputNetworkData activation,
                                OutputNetworkData target) = 0;
      double summation(OutputNetworkData activations, OutputNetworkData targets,
                       double (*ecuation)(double, double));
      virtual ~LossFunction() = default;
};

struct MeanSquaredError : public LossFunction {
      double function(OutputNetworkData activations,
                      OutputNetworkData targets) override {
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
                  summ += (2.0) * (target[it] - activation[it]);
            }
            return summ / (N);
      }
};

struct BinaryCrossEntropy : public LossFunction {
      double function(OutputNetworkData activations,
                      OutputNetworkData targets) override {
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
                  if (activation[it] <= 1e-15)
                        activation[it] = 1e-15;
                  else if (activation[it] >= 1 - 1e-15)
                        activation[it] = 1 - 1e-15;
                  summ += (activation[it] - target[it]) /
                          (activation[it] * (1 - activation[it]));
            }
            return (summ / N);
      }
};

std::shared_ptr<LossFunction> newInstance(TYPE type);
} // namespace LossFuctions
#endif
