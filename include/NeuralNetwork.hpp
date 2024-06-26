#ifndef __NEURALNETWORK_H__
#define __NEURALNETWORK_H__

#include "Data.hpp"
#include "LossFuctions.hpp"
#include "NetworkGlobalResources.hpp"
#include "NeuralNetworkImpl.hpp"
#include <memory>
class NeuralNetwork {
    public:
      using NetworkDescription = NeuralNetworkImpl::NetworkDescription;
      NeuralNetwork(NetworkDescription networkDescription,
                    LossFuctions::TYPE lossFunctionType, double initialAlpha);
      void fit(NetworkTrainData trainData, int epochs, int batchSize);
      OutputNetworkData predict(InputNetworkData input);
      void addWarmUp(double initialAlpha, double finalAlpha, int periodEpoch);
      void addDecayLearningRate(double initialAlpha, double finalAlpha,
                                int periodEpoch);

    private:
      NeuralNetworkImpl net_impl;
};

#endif
