#ifndef __NEURALNETWORK_H__
#define __NEURALNETWORK_H__

#include "Data.hpp"
#include "NeuralNetworkImpl.hpp"
class NeuralNetwork {
    public:
      using NetworkDescription = NeuralNetworkImpl::NetworkDescription;
      NeuralNetwork(NetworkDescription networkDescription);
      void fit(NetworkTrainData trainData, int epochs, int batchSize);
      OutputNetworkData predict(InputNetworkData input);

    private:
      NeuralNetworkImpl net_impl;
};

#endif
