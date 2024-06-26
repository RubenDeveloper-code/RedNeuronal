#include "../include/NeuralNetwork.hpp"
#include "../include/Data.hpp"
#include "../include/NeuralNetworkFit.hpp"
#include "../include/NeuralNetworkImpl.hpp"
#include <vector>

NeuralNetwork::NeuralNetwork(NetworkDescription networkDescription,
                             LossFuctions::TYPE lossFunctionType,
                             double initialAlpha)
    : net_impl{networkDescription, lossFunctionType, {initialAlpha}} {}

void NeuralNetwork::fit(NetworkTrainData trainData, int epochs, int batchSize) {
      NeuralNetworkFit net_fit(trainData, batchSize, epochs, &net_impl);
      net_fit.fit();
}
void NeuralNetwork::addWarmUp(double initialAlpha, double finalAlpha,
                              int limitEpochs) {
      net_impl.netAlgorithmsAlpha.upWarmUp(std::move(
          NetworkAlgorithms::WarmUp{initialAlpha, finalAlpha, limitEpochs}));
}
void NeuralNetwork::addDecayLearningRate(double initialAlpha, double finalAlpha,
                                         int limitEpochs) {
      net_impl.netAlgorithmsAlpha.upDecayLearningRate(
          std::move(NetworkAlgorithms::DecayLearningRate{
              initialAlpha, finalAlpha, limitEpochs}));
}

std::vector<double> NeuralNetwork::predict(InputNetworkData input) {
      if (input.size() != net_impl.getInputSize())
            return std::vector<double>{-1};
      return net_impl._predict(input);
}
