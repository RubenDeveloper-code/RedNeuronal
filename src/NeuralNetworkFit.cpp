#include "../include/NeuralNetworkFit.hpp"
#include "../include/Data.hpp"
#include "../include/NeuralNetworkImpl.hpp"
#include "../include/NeuralNetworkSetterData.hpp"
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

NeuralNetworkFit::NeuralNetworkFit(NetworkTrainData trainData, int _batchSize,
                                   int _epochs, std::shared_ptr<int> epochs_ptr,
                                   NeuralNetworkImpl *impl)
    : mini_batch(_batchSize), epochs(_epochs), net_impl(impl),
      actualEpoch(epochs_ptr) {
      setterData = SetterData(trainData);
}

// minibatch default 1
void NeuralNetworkFit::fit() {
      std::vector<OutputNetworkData> computedOutputs;
      std::vector<OutputNetworkData> desiredOutputs;
      while ((*actualEpoch)++ < epochs) {
            //  invariante, podria considerar los que queden fuera del multiplo
            for (int multi = 1; multi * mini_batch <= setterData.getDataSize();
                 multi++) {
                  for (auto i = 0; i < mini_batch; i++) {
                        auto newData = prepareEpoch();
                        desiredOutputs.push_back(newData.output);
                        computedOutputs.push_back(computeOutput());
                  }
                  net_impl->recalculateWeights(computedOutputs, desiredOutputs);
                  calculeLoss(computedOutputs, desiredOutputs, epochs);
                  computedOutputs.clear();
                  desiredOutputs.clear();
            }
      }
}

Data NeuralNetworkFit::prepareEpoch() {
      return setterData.prepareNextEpoch(net_impl->getInputLayer(),
                                         net_impl->getOutputLayer());
}

OutputNetworkData NeuralNetworkFit::computeOutput() {
      return net_impl->generateOutput();
}

double
NeuralNetworkFit::calculeLoss(std::vector<OutputNetworkData> desiredOutputs,
                              std::vector<OutputNetworkData> computedOutputs,
                              int epochs) {
      double loss =
          net_impl->lossFunction->function(desiredOutputs, computedOutputs);
      showLoss(loss, epochs);
      return loss;
}

void NeuralNetworkFit::showLoss(long loss, int epochs) {
      static int cont = 0;
      int call = (epochs / mini_batch) / 10;
      if (cont++ == call) {
            if (loss < 10) {
                  /*for (int i = 0; i < loss * 10; i++) {
                        std::cout << "█";
                  }*/
            }
            cont = 0;
            std::cout << "::: " << loss << " in epoch: " << *actualEpoch
                      << " \n";
      }
}
