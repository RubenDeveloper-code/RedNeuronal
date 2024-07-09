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
      enum class LossFuctions { MSE, BCE } loss_function;
      LayerDesign(LayerClass tp, Activations act, Optimizers opt, int nNeurons)
          : type(tp), activation(act), optimizer(opt), n_neurons(nNeurons){};
      void checkIntegrity();
};

#endif
