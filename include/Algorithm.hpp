#ifndef __ALGORITMS_HPP__
#define __ALGORITMS_HPP__

#include <cmath>
#include <memory>
namespace Algorithms {
enum class TYPE { SGD, ADAMS };
struct NeuronConectionInfo {
      NeuronConectionInfo(double _val, double _alpha, long double _gradient)
          : val(_val), alpha(_alpha), gradient(_gradient) {}
      double val;
      double alpha;
      double gradient;
};
struct OptimizationAlgorithm {
      virtual double optimizeWeigth(NeuronConectionInfo context) = 0;
      virtual double optimizeBias(NeuronConectionInfo context) = 0;
      virtual ~OptimizationAlgorithm() = default;
};
struct SDG : public OptimizationAlgorithm {
      double optimizeWeigth(NeuronConectionInfo context) override {
            return context.val - context.alpha * context.gradient;
      }
      double optimizeBias(NeuronConectionInfo context) override {
            return context.val - context.alpha * context.gradient;
      };
};

struct Adams : public OptimizationAlgorithm {
      struct hiperparameters {
            double beta1 = 0.9, beta2 = 0.999;
            double epsilon = 10e-9;
            double m = 0, v = 0;
            double mC{}, vC{};
            double gradient{};
            double alpha = 0.01;
      };
      hiperparameters biashp;
      hiperparameters weighthp;
      std::shared_ptr<int> t{};
      Adams(std::shared_ptr<int> _t) : t{_t} {};
      double optimizeWeigth(NeuronConectionInfo context) override {
            return computeParameter(context, weighthp);
      }
      double optimizeBias(NeuronConectionInfo context) override {
            return computeParameter(context, biashp);
      }

      double computeParameter(NeuronConectionInfo context,
                              hiperparameters &data) {
            computeFirstFixedMomentum(data);
            computeSecondFixedMomentum(data);
            return context.val -
                   data.alpha * (data.mC / (sqrt(data.vC) + data.epsilon));
      }
      void computeFirstFixedMomentum(hiperparameters &data) {
            data.m = data.beta1 * data.m + (1 - data.beta1) * data.gradient;
            data.mC = data.m / (1 - pow(data.beta1, *t));
      }
      void computeSecondFixedMomentum(hiperparameters &data) {
            data.v = data.beta2 * data.v +
                     (1 - data.beta2) * std::pow(data.gradient, 2);
            data.vC = data.v / (1 - pow(data.beta2, *t));
      }
};

inline std::shared_ptr<OptimizationAlgorithm>
newInstance(TYPE type, std::shared_ptr<int> t = nullptr) {
      switch (type) {
      case TYPE::SGD:
            return std::make_shared<SDG>();
      case TYPE::ADAMS:
            return std::make_shared<Adams>(t);
      }
      return std::make_shared<SDG>();
}
} // namespace Algorithms
#endif
