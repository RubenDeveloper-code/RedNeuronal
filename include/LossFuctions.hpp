#ifndef __LOSSFUNCTIONS_HPP__
#define __LOSSFUNCTIONS_HPP__
#include "Data.hpp"
#include <cmath>
#include <memory>

namespace LossFuctions {
enum class TYPE { MSE };
struct LossFunction {
      virtual double function(OutputNetworkData activations,
                              OutputNetworkData targets) = 0;
      virtual double derivative(double activation, double target) = 0;
      virtual ~LossFunction() = default;
};
struct MeanSquaredError : public LossFunction {
      double function(OutputNetworkData activations,
                      OutputNetworkData targets) override {
            const int N = activations.size();
            long summation{};
            long loss{};
            for (int i = 0; i < N; i++) {
                  summation += std::pow(targets[i] - activations[i], 2);
            }
            loss = summation / 2;
            return loss;
      }
      double derivative(double activation, double target) override {
            return (target - activation);
      }
};
inline std::shared_ptr<LossFunction> newInstance(TYPE type) {
      switch (type) {
      case TYPE::MSE:
            return std::make_shared<MeanSquaredError>();
      }
      return std::make_shared<MeanSquaredError>();
}
} // namespace LossFuctions
#endif
