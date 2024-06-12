#ifndef __NEURALNETWORK_FIT_HPP__
#define __NEURALNETWORK_FIT_HPP__

#include "Data.hpp"
#include "NeuralNetworkImpl.hpp"
#include "NeuralNetworkSetterData.hpp"
#include <vector>

class NeuralNetworkFit {
    public:
      NeuralNetworkFit(NetworkTrainData, int epochs, int batchSize,
                       NeuralNetworkImpl *impl);
      void fit();

    private:
      SetterData setterData;
      NeuralNetworkImpl *net_impl;
      int actualEpoch, epochs, batchSize;
      std::vector<double> batchLoss{};
      int batch_ind = 0;

      Data prepareEpoch();
      OutputNetworkData stepTrain();
      double calculeLoss(OutputNetworkData target, OutputNetworkData out,
                         int epochs);
      void showLoss(std::vector<double> batchLoss, int epochs);
};

#endif
