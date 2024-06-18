#ifndef __ALGORITMS_HPP__
#define __ALGORITMS_HPP__

#include <cmath>
#include <memory>
namespace Algorithms {
enum class TYPE { SGD, ADAMS };
struct NeuronConectionInfo {
      NeuronConectionInfo(double _val, double _alpha, double _delta,
                          double _neuronValue_p = 0)
          : val(_val), alpha(_alpha), delta(_delta),
            neuronValue_p(_neuronValue_p){};
      double val;
      double alpha;
      double delta;
      double neuronValue_p;
};
struct OptimizationAlgorithm {
      virtual double optimizeWeigth(NeuronConectionInfo context) = 0;
      virtual double optimizeBias(NeuronConectionInfo context) = 0;
      virtual ~OptimizationAlgorithm() = default;
};
struct SDG : public OptimizationAlgorithm {
      double optimizeWeigth(NeuronConectionInfo context) override {
            return context.val -
                   context.alpha * context.delta * -context.neuronValue_p;
      }
      double optimizeBias(NeuronConectionInfo context) override {
            return context.val - context.alpha * context.delta * -1;
      };
};

struct Adams : public OptimizationAlgorithm {
      double beta1 = 0.9, beta2 = 0.999;
      double epsilon = 10e-9;
      double m = 0, v = 0;
      double mC{}, vC{};
      double gradient{};
      std::shared_ptr<int> t{};
      Adams(std::shared_ptr<int> _t) : t{_t} {};
      double optimizeWeigth(NeuronConectionInfo context) override {
            return computeParameter(context);
      }
      double optimizeBias(NeuronConectionInfo context) override {
            return computeParameter(context);
      }

      double computeParameter(NeuronConectionInfo context) {
            gradient = context.delta * context.neuronValue_p;
            computeFirstFixedMomentum();
            computeSecondFixedMomentum();
            return context.val - context.alpha * (mC / sqrt(vC) + epsilon);
      }
      void computeFirstFixedMomentum() {
            m = beta1 * m + (1 - beta1) * gradient;
            mC = std::pow(m / 1 - beta1, *t);
      }
      void computeSecondFixedMomentum() {
            v = beta2 * v + (1 - beta2) * std::pow(gradient, 2);
            vC = std::pow(v / 1 - beta2, *t);
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
