#ifndef __ALGORITMS_HPP__
#define __ALGORITMS_HPP__

#include <cmath>
#include <iostream>
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
            double epsilon = 10e-8;
            double m = 0, v = 0;
            double mC{}, vC{};
            double alpha = 0.1;
      };
      hiperparameters biashp;
      hiperparameters weighthp;
      std::shared_ptr<int> t{};
      Adams(std::shared_ptr<int> _t) : t{_t}, biashp(), weighthp(){};
      double optimizeWeigth(NeuronConectionInfo context) override {
            return computeParameter(context, weighthp);
      }
      double optimizeBias(NeuronConectionInfo context) override {
            return computeParameter(context, biashp);
      }

      double computeParameter(NeuronConectionInfo context,
                              hiperparameters &data) {
            computeFirstFixedMomentum(data, context.gradient);
            computeSecondFixedMomentum(data, context.gradient);
            return context.val -
                   data.alpha * (data.mC / (sqrt(data.vC) + data.epsilon));
      }
      void computeFirstFixedMomentum(hiperparameters &data, double gradient) {
            data.m = data.beta1 * data.m + (1.0 - data.beta1) * gradient;
            data.mC = data.m / (1.0 - pow(data.beta1, *t));
      }
      void computeSecondFixedMomentum(hiperparameters &data, double gradient) {
            data.v = data.beta2 * data.v +
                     (1.0 - data.beta2) * std::pow(gradient, 2.0);
            data.vC = data.v / (1.0 - pow(data.beta2, *t));
      }
      void normalizeGradient(NeuronConectionInfo &context) {
            double norm = sqrt(context.gradient * context.gradient);
            if (norm > 0)
                  context.gradient /= norm;
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
