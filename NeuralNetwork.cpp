#include "NeuralNetwork.hpp"
#include "Data.hpp"
#include "NeuralNetworkFit.hpp"
#include "NeuralNetworkImpl.hpp"
#include <vector>

NeuralNetwork::NeuralNetwork(NetworkDescription networkDescription)
    : net_impl(networkDescription) {}

void NeuralNetwork::fit(NetworkTrainData trainData, int epochs, int batchSize) {
      NeuralNetworkFit net_fit(trainData, epochs, batchSize, &net_impl);
      net_fit.fit();
}

std::vector<double> NeuralNetwork::predict(InputNetworkData input) {
      if (input.size() != net_impl.getInputSize())
            return std::vector<double>{-1};
      return net_impl._predict(input);
}
