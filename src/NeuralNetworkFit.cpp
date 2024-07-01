#include "../include/NeuralNetworkFit.hpp"
#include "../include/Data.hpp"
#include "../include/NeuralNetworkImpl.hpp"
#include "../include/NeuralNetworkSetterData.hpp"
#include <iostream>
#include <memory>
#include <vector>

NeuralNetworkFit::NeuralNetworkFit(NetworkTrainData trainData, int _batchSize,
                                   int _epochs, double _deadfitline,
                                   NeuralNetworkImpl *impl)
    : mini_batch(_batchSize), epochs(_epochs), deadfitline(_deadfitline),
      net_impl(impl) {
      setterData = SetterData(trainData);
}

void NeuralNetworkFit::fit() {
      std::vector<OutputNetworkData> computedOutputs;
      std::vector<OutputNetworkData> desiredOutputs;
      net_impl->initAlphaAlgorithms();
      // con el tamaño del minibatch
      double summLoss{};
      while ((*net_impl->GLOBAL_RESOURSES.epochs_it)++ < epochs) {
            net_impl->algorithmsAlpha->run();
            for (int multi = 1;
                 multi * mini_batch < setterData.getDataSize() + mini_batch;
                 multi++) {
                  for (auto i = 0; i < mini_batch; i++) {
                        auto newData = prepareEpoch();
                        desiredOutputs.push_back(newData.output);
                        computedOutputs.push_back(computeOutput());
                  }
                  net_impl->recalculateWeights(computedOutputs, desiredOutputs);
                  double loss =
                      calculeLoss(computedOutputs, desiredOutputs, epochs);
                  summLoss += loss;
                  computedOutputs.clear();
                  desiredOutputs.clear();
            }
            if (summLoss / mini_batch < deadfitline)
                  return;
            summLoss = 0;
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

void NeuralNetworkFit::showLoss(double loss, int epochs) {
      static int cont = 0;
      int call = (epochs / mini_batch) / 10;
      if (cont++ == call) {
            if (loss < 10) {
                  /*for (int i = 0; i < loss * 10; i++) {
                        std::cout << "█";
                  }*/
            }
            cont = 0;
            std::cout << "::: " << loss
                      << " in epoch: " << *net_impl->GLOBAL_RESOURSES.epochs_it
                      << " \n";
      }
}
