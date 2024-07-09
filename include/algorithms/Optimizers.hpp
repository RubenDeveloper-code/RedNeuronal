#ifndef __ALGORITMS_HPP__
#define __ALGORITMS_HPP__

#include "../core/SharedResources.hpp"
#include <cmath>
#include <memory>
namespace Optimizers {
enum class TYPE { SGD, ADAMS, ADAGRAD, RMSPROP, MOMENTUM };
struct NeuronConectionInfo {
      NeuronConectionInfo(double _val, long double _gradient)
          : val(_val), gradient(_gradient) {}
      double val;
      double gradient;
};
struct Optimizer {
      Optimizer(std::shared_ptr<double> _alpha) : alpha(_alpha) {}
      std::shared_ptr<double> alpha;
      virtual double optimizeWeigth(NeuronConectionInfo context) = 0;
      virtual double optimizeBias(NeuronConectionInfo context) = 0;
      double L2Gradient(double theta, double gradient) {
            double lada = 10e-1;
            return gradient;
            return gradient + theta * lada;
      }
      virtual ~Optimizer() = default;
};
struct SDG : public Optimizer {
      SDG(std::shared_ptr<double> _alpha) : Optimizer(_alpha){};
      double optimizeWeigth(NeuronConectionInfo context) override {
            // return context.val -
            //      *alpha * (L2Gradient(context.val, context.gradient));
            return context.val - *alpha * context.gradient;
      }
      double optimizeBias(NeuronConectionInfo context) override {
            // return context.val -
            //      *alpha * (L2Gradient(context.val, context.gradient));
            return context.val - *alpha * context.gradient;
      };
};

struct Adams : public Optimizer {
      struct hiperparameters {
            double beta1 = 0.9, beta2 = 0.999;
            double epsilon = 10e-8;
            double m = 0, v = 0;
            double mC{}, vC{};
      };
      hiperparameters biashp;
      hiperparameters weighthp;
      std::shared_ptr<int> t{};
      Adams(std::shared_ptr<int> _t, std::shared_ptr<double> alpha)
          : Optimizer(alpha), t{_t}, biashp(), weighthp(){};

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
                   *alpha * (data.mC / (sqrt(data.vC) + data.epsilon));
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

struct AdaGrad : public Optimizer {
      struct Hiperparameters {
            double G{};
            double epsilon = 10e-8;
      };
      AdaGrad(std::shared_ptr<double> alpha) : Optimizer(alpha){};
      Hiperparameters biashp, weighthp;
      double optimizeWeigth(NeuronConectionInfo context) override {
            computeG(weighthp, context.gradient);
            return context.val -
                   (*alpha / (sqrt(weighthp.G) + weighthp.epsilon)) *
                       context.gradient;
      }
      double optimizeBias(NeuronConectionInfo context) override {
            computeG(biashp, context.gradient);
            return context.val - (*alpha / (sqrt(biashp.G) + biashp.epsilon)) *
                                     context.gradient;
      }

      void computeG(Hiperparameters &hiperparameters, double gradient) {
            hiperparameters.G = hiperparameters.G + std::pow(gradient, 2.0);
      }
};

struct RMSProp : public Optimizer {
      struct Hiperparameters {
            double Eg{};
            double epsilon = 10e-8;
            double beta = 0.9;
      };

      RMSProp(std::shared_ptr<double> alpha) : Optimizer(alpha){};
      Hiperparameters biashp, weighthp;
      double optimizeWeigth(NeuronConectionInfo context) override {
            computeEg(weighthp, context.gradient);
            return context.val -
                   (*alpha / (sqrt(weighthp.Eg) + weighthp.epsilon)) *
                       context.gradient;
      }
      double optimizeBias(NeuronConectionInfo context) override {
            computeEg(biashp, context.gradient);
            return context.val - (*alpha / (sqrt(biashp.Eg) + biashp.epsilon)) *
                                     context.gradient;
      }

      void computeEg(Hiperparameters &hiperparameters, double gradient) {
            hiperparameters.Eg =
                hiperparameters.beta * hiperparameters.Eg +
                (1 - hiperparameters.beta) * std::pow(gradient, 2.0);
      }
};

struct Momentum : public Optimizer {
      struct Hiperparameters {
            double v = 0.0;
            double beta = 0.9;
      };

      Momentum(std::shared_ptr<double> alpha) : Optimizer(alpha){};
      Hiperparameters biashp, weighthp;
      double optimizeWeigth(NeuronConectionInfo context) override {
            computeV(weighthp, context.gradient);
            return context.val - (*alpha * weighthp.v);
      }
      double optimizeBias(NeuronConectionInfo context) override {
            computeV(biashp, context.gradient);
            return context.val - (*alpha * biashp.v);
      }

      void computeV(Hiperparameters &hiperparameters, double gradient) {
            hiperparameters.v = hiperparameters.beta * hiperparameters.v +
                                (1 - hiperparameters.beta) * gradient;
      }
};

std::shared_ptr<Optimizer> newInstance(TYPE type,
                                       SharedResources &shared_resources);
} // namespace Optimizers
#endif
