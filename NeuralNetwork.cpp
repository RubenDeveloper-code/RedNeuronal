#include "NeuralNetwork.hpp"
#include "NeuralNetwork_impl.hpp"
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
#include <ostream>
#include <vector>

NeuralNetwork::NeuralNetwork(NetworkDimentions netDims,
                             NeuronActivations::activation *wideActivation,
                             NeuronActivations::activation *outActivation)
    : net_impl(netDims, wideActivation, outActivation) {}

void NeuralNetwork::fit(NeuralNetwork_impl::NetworkData inputData,
                        NeuralNetwork_impl::NetworkData outputData, int epochs,
                        int lossStep) {
      NeuralNetwork_fit net_fit(inputData, outputData, epochs, lossStep,
                                &net_impl);
      net_fit.fit();
}

// el tipo de dato variara
std::vector<double>
NeuralNetwork::predict(NeuralNetwork_impl::NetworkData input) {
      if (input.size() != net_impl.getInputSize())
            return std::vector<double>{-1};
      return net_impl._predict(input);
}
