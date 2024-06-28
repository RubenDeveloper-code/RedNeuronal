#ifndef __NEURALNETWORK_H__
#define __NEURALNETWORK_H__

#include "Data.hpp"
#include "LossFuctions.hpp"
#include "NetworkAlgoritms.hpp"
#include "NetworkGlobalResources.hpp"
#include "NeuralNetworkImpl.hpp"
#include <memory>
class NeuralNetwork {
    public:
      using NetworkDescription = NeuralNetworkImpl::NetworkDescription;
      NeuralNetwork(NetworkDescription networkDescription,
                    LossFuctions::TYPE lossFunctionType, double initialAlpha);
      void fit(NetworkTrainData trainData, int epochs, int batchSize,
               double deadfitline);
      OutputNetworkData predict(InputNetworkData input);
      NetworkAlgorithms::AlgorithmsAlpha alphaAlgorithms;

    private:
      NeuralNetworkImpl net_impl;
};

#endif
