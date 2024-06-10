#ifndef __NEURALNETWORK_H__
#define __NEURALNETWORK_H__

#include "NeuralNetwork_fit.hpp"
#include "NeuralNetwork_impl.hpp"
#include "Neuron.hpp"
#include <memory>
#include <vector>
class NeuralNetwork {
    public:
      NeuralNetwork(NetworkDimentions netDims,
                    NeuronActivations::activation *wideLayer,
                    NeuronActivations::activation *outLayer);
      void fit(NeuralNetwork_impl::NetworkData input,
               NeuralNetwork_impl::NetworkData output, int epochs,
               int lossStep);
      std::vector<double> predict(NeuralNetwork_impl::NetworkData input);

    private:
      NeuralNetwork_impl net_impl;
};

#endif
