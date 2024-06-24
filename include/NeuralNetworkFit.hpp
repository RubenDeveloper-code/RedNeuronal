#ifndef __NEURALNETWORK_FIT_HPP__
#define __NEURALNETWORK_FIT_HPP__

#include "Data.hpp"
#include "NeuralNetworkImpl.hpp"
#include "NeuralNetworkSetterData.hpp"
#include <memory>
#include <vector>

class NeuralNetworkFit {
    public:
      NeuralNetworkFit(NetworkTrainData, int mini_batch, int epochs,
                       std::shared_ptr<int> epoch_ptr, NeuralNetworkImpl *impl);
      void fit();

    private:
      SetterData setterData;
      NeuralNetworkImpl *net_impl;
      std::shared_ptr<int> actualEpoch;
      int epochs, mini_batch;
      ;
      std::vector<double> batchLoss{};
      int batch_ind = 0;

      Data prepareEpoch();
      OutputNetworkData computeOutput();
      double calculeLoss(std::vector<OutputNetworkData> desiredOutputs,
                         std::vector<OutputNetworkData> computedOutputs,
                         int epochs);
      void showLoss(long loss, int epochs);
};

#endif
