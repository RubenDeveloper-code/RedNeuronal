#ifndef __LAYER_HPP__
#define __LAYER_HPP__

#include "../../designs/LayerDesign.hpp"
#include "../resources/SharedResources.hpp"
#include "Neuron.hpp"
#include <vector>
class Layer {
    public:
      enum class TYPE { INPUT, HIDE, OUTPUT };
      Layer(LayerDesign,
            std::shared_ptr<LossFuctions::LossFunction> loss_function,
            SharedResources &shared_resources);
      std::vector<Neuron> neurons;
      TYPE type;
      std::vector<Neuron>::iterator begin() { return neurons.begin(); }
      std::vector<Neuron>::iterator end() { return neurons.end(); }
};

#endif
