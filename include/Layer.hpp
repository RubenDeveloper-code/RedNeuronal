#ifndef __LAYER_HPP__
#define __LAYER_HPP__

#include "Algorithm.hpp"
#include "LossFuctions.hpp"
#include "Neuron.hpp"
#include "NeuronActivation.hpp"
#include <memory>

struct LayerDescription {
      LayerDescription(Neuron::TYPE _type, NeuronActivations::TYPE _activation,
                       Algorithms::TYPE _algorithm, int _nNeurons)
          : type(_type), activation(_activation), algorithm(_algorithm),
            nNeurons(_nNeurons){};
      Neuron::TYPE type;
      NeuronActivations::TYPE activation;
      Algorithms::TYPE algorithm;
      int nNeurons;
};
struct Layer {
      Layer(){};
      Layer(Neuron::TYPE _type, NeuronActivations::TYPE activation,
            Algorithms::TYPE algorithm,
            std::shared_ptr<LossFuctions::LossFunction> lossFunction,
            std::shared_ptr<int> epoch_ptr, int _nNeurons)
          : type{_type}, nNeurons(_nNeurons) {
            while (_nNeurons-- > 0) {
                  neurons.emplace_back(
                      Neuron{NeuronActivations::newInstance(activation),
                             Algorithms::newInstance(algorithm, epoch_ptr),
                             lossFunction, type});
            }
      }
      Neuron::TYPE type;
      Neuron::Neurons neurons;
      int nNeurons;
};

#endif
