#ifndef __ALGORITMS_HPP__
#define __ALGORITMS_HPP__

#include <memory>
namespace Algorithms {
enum class TYPE { DEFAULT, ADAMS };
struct NeuronConectionInfo {
      NeuronConectionInfo(double _alpha, double _delta, double _val,
                          double _neuronValue_p = 0)
          : alpha(_alpha), delta(_delta), val(_val),
            neuronValue_p(_neuronValue_p){};
      double alpha;
      double delta;
      double val;
      double neuronValue_p;
};
struct OptimizationAlgorithm {
      virtual double optimizeWeigth(NeuronConectionInfo context) = 0;
      virtual double optimizeBias(NeuronConectionInfo context) = 0;
};
struct Default : public OptimizationAlgorithm {
      double optimizeWeigth(NeuronConectionInfo context) override {
            return -context.val * context.alpha * context.delta *
                   context.neuronValue_p;
      }
      virtual double optimizeBias(NeuronConectionInfo context) override {
            return -context.val * context.alpha * context.delta;
      };
};
inline std::shared_ptr<OptimizationAlgorithm> newInstance(TYPE type) {
      switch (type) {
      case TYPE::DEFAULT:
            return std::make_shared<Default>();
      case TYPE::ADAMS:
            return std::make_shared<Default>();
      }
      return std::make_shared<Default>();
}
} // namespace Algorithms
#endif
