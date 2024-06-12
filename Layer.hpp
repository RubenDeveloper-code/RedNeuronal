#ifndef __LAYER_HPP__
#define __LAYER_HPP__

#include "Algorithm.hpp"
#include "Neuron.hpp"
#include "NeuronActivation.hpp"

// una funcion retornara una instacia acorde a el tipo seleccionado por el enum
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
            Algorithms::TYPE algorithm, int _nNeurons)
          : type{_type}, nNeurons(_nNeurons) {
            while (_nNeurons-- > 0) {
                  neurons.emplace_back(
                      Neuron{NeuronActivations::newInstance(activation),
                             Algorithms::newInstance(algorithm), _type});
            }
      }
      Neuron::TYPE type;
      Neuron::Neurons neurons;
      int nNeurons;
};

#endif
