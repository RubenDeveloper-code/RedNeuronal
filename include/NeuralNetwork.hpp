#ifndef __NEURALNETWORK_H__
#define __NEURALNETWORK_H__

#include "Data.hpp"
#include "LossFuctions.hpp"
#include "NeuralNetworkImpl.hpp"
#include <memory>
class NeuralNetwork {
    public:
      using NetworkDescription = NeuralNetworkImpl::NetworkDescription;
      NeuralNetwork(NetworkDescription networkDescription,
                    LossFuctions::TYPE lossFunctionType);
      void fit(NetworkTrainData trainData, int epochs, int batchSize);
      OutputNetworkData predict(InputNetworkData input);

    private:
      NeuralNetworkImpl net_impl;
};

#endif
