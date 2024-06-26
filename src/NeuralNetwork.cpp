#include "../include/NeuralNetwork.hpp"
#include "../include/Data.hpp"
#include "../include/NeuralNetworkFit.hpp"
#include "../include/NeuralNetworkImpl.hpp"
#include <vector>

NeuralNetwork::NeuralNetwork(NetworkDescription networkDescription,
                             LossFuctions::TYPE lossFunctionType,
                             double initialAlpha)
    : alphaAlgorithms(), net_impl{networkDescription,
                                  lossFunctionType,
                                  {initialAlpha},
                                  alphaAlgorithms} {}

void NeuralNetwork::fit(NetworkTrainData trainData, int epochs, int batchSize) {
      NeuralNetworkFit net_fit(trainData, batchSize, epochs, &net_impl);
      net_fit.fit();
}

std::vector<double> NeuralNetwork::predict(InputNetworkData input) {
      if (input.size() != net_impl.getInputSize())
            return std::vector<double>{-1};
      return net_impl._predict(input);
}
