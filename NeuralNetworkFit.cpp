#include "NeuralNetworkFit.hpp"
#include "Data.hpp"
#include "NeuralNetworkImpl.hpp"
#include "NeuralNetworkSetterData.hpp"
#include <iostream>
#include <numeric>

NeuralNetworkFit::NeuralNetworkFit(NetworkTrainData trainData, int _epochs,
                                   int _batchSize, NeuralNetworkImpl *impl)
    : batchSize(_batchSize), actualEpoch(), epochs(_epochs), net_impl(impl) {
      setterData = SetterData(trainData);
}

void NeuralNetworkFit::fit() {
      while (actualEpoch++ < epochs) {
            auto newData = prepareEpoch();
            OutputNetworkData out = stepTrain();
            net_impl->recalculateWeights();
            calculeLoss(out, newData.output, epochs);
      }
}

Data NeuralNetworkFit::prepareEpoch() {
      return setterData.prepareNextEpoch(NeuralNetworkImpl::input,
                                         NeuralNetworkImpl::output);
}

OutputNetworkData NeuralNetworkFit::stepTrain() {
      return net_impl->generateOutput();
}

double NeuralNetworkFit::calculeLoss(OutputNetworkData target,
                                     OutputNetworkData out, int epochs) {
      int n = out.size();
      double loss = 0;
      for (int i = 0; i < n; i++) {
            loss += std::abs(out[i] - target[i]);
      }
      loss /= n;
      if (batch_ind++ == batchSize) {
            batch_ind = 0;
            showLoss(batchLoss, epochs);
            batchLoss.clear();
      } else
            batchLoss.push_back(loss);
      return loss;
}

void NeuralNetworkFit::showLoss(std::vector<double> batchLoss, int epochs) {
      int div = epochs / 10;
      double _loss =
          std::accumulate(batchLoss.begin(), batchLoss.end(), 0.0f) / batchSize;
      if ((actualEpoch % div) == 0) {
            if (_loss < 10) {
                  for (int i = 0; i < _loss * 10; i++) {
                        std::cout << "█";
                  }
            }
            std::cout << " " << _loss << " in epoch: " << actualEpoch << " \n";
      }
}
