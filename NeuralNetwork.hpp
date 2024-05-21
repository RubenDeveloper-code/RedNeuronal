#ifndef __NEURALNETWORK_H__
#define __NEURALNETWORK_H__

#include "NeuralNetwork_impl.hpp"
#include "Neuron.hpp"
#include <vector>
class NeuralNetwork {
    public:
      NeuralNetwork(NetworkDimentions netDims, Neuron::Activation activation);
      void fit(NeuralNetwork_impl::NetworkData input,
               NeuralNetwork_impl::NetworkData output, int epochs);

    private:
      NeuralNetwork_impl net_impl;
};

#endif
