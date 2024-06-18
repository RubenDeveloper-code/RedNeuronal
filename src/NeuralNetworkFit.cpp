#include "../include/NeuralNetworkFit.hpp"
#include "../include/Data.hpp"
#include "../include/NeuralNetworkImpl.hpp"
#include "../include/NeuralNetworkSetterData.hpp"
#include <iostream>
#include <memory>
#include <numeric>

NeuralNetworkFit::NeuralNetworkFit(NetworkTrainData trainData, int _epochs,
                                   int _batchSize, NeuralNetworkImpl *impl)
    : batchSize(_batchSize), epochs(_epochs), net_impl(impl) {
      setterData = SetterData(trainData);
}

void NeuralNetworkFit::fit() {
      while ((actualEpoch)++ < epochs) {
            auto newData = prepareEpoch();
            OutputNetworkData out = stepTrain();
            net_impl->recalculateWeights();
            calculeLoss(out, newData.output, epochs);
      }
}

Data NeuralNetworkFit::prepareEpoch() {
      return setterData.prepareNextEpoch(net_impl->getInputLayer(),
                                         net_impl->getOutputLayer());
}

OutputNetworkData NeuralNetworkFit::stepTrain() {
      return net_impl->generateOutput();
}

double NeuralNetworkFit::calculeLoss(OutputNetworkData target,
                                     OutputNetworkData out, int epochs) {
      double loss = net_impl->lossFunction->function(out, target);
      if (batch_ind++ == batchSize) {
            batch_ind = {};
            showLoss(batchLoss, epochs);
            batchLoss.clear();
      } else
            batchLoss.push_back(loss);
      return loss;
}

void NeuralNetworkFit::showLoss(std::vector<double> batchLoss, int epochs) {
      static int cont = 0;
      int call = (epochs / batchSize) / 10;
      double loss =
          std::accumulate(batchLoss.begin(), batchLoss.end(), 0.0f) / batchSize;
      if (cont++ == call) {
            if (loss < 10) {
                  for (int i = 0; i < loss * 10; i++) {
                        std::cout << "█";
                  }
            }
            cont = 0;
            std::cout << "::: " << loss << " in epoch: " << actualEpoch
                      << " \n";
      }
}
