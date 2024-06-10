#ifndef __NEURON_ACTVATION_H__
#define __NEURON_ACTVATION_H__

#include <algorithm>
#include <cmath>
namespace NeuronActivations {
struct activation {
      double (*function)(double) = nullptr;
      double (*derivative)(double, double) = nullptr;
      virtual double getDevStandart(double seed) = 0;
      virtual ~activation() = default;
};
struct sigmoid : public activation {
      sigmoid() {
            function = [](double x) { return (1.0 / (1.0 + std::exp(-x))); };
            derivative = [](double x, double y) { return (y * (1.0 - y)); };
      }
      double getDevStandart(double seed) { return std::sqrt(1.0 / seed); }
};
struct relu : public activation {
      relu() {
            function = [](double x) { return (std::max(0.0, x)); };
            derivative = [](double x, double y) { return (x > 0) ? 1.0 : 0.0; };
      }
      double getDevStandart(double seed) { return std::sqrt(2.0 / seed); }
};
struct regression : public activation {
      regression() {
            function = [](double x) { return x; };
            derivative = [](double x, double y) { return 1.0; };
      }
      double getDevStandart(double seed) { return std::sqrt(2.0 / seed); }
};
} // namespace NeuronActivations

#endif
