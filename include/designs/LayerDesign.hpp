#ifndef __LAYER_DESIGN_HPP__
#define __LAYER_DESIGN_HPP__

#include "../algorithms/LossFuctions.hpp"
#include <memory>
struct LayerDesign {
      enum class LayerClass { INPUT, HIDE, OUTPUT } type;
      enum class Activations { SIGMOID, RELU, REGRESSION } activation;
      enum class Optimizers {
            SGD,
            ADAMS,
            ADAGRAD,
            RMSPROP,
            MOMENTUM
      } optimizer;
      int n_neurons;
      double dropout_p;
      enum class LossFuctions { MSE, BCE } loss_function;
      LayerDesign(LayerClass tp, Activations act, Optimizers opt, int nNeurons,
                  double dropout_p = 0)
          : type(tp), activation(act), optimizer(opt), n_neurons(nNeurons),
            dropout_p(dropout_p){};
      void checkIntegrity();
};

#endif
