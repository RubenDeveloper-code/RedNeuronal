#include "../include/NeuralNetwork.hpp"
#include "../include/Data.hpp"
#include "../include/NeuralNetworkFit.hpp"
#include "../include/NeuralNetworkImpl.hpp"
#include <vector>

NeuralNetwork::NeuralNetwork(NetworkDescription networkDescription,
                             LossFuctions::TYPE lossFunctionType) {
      net_impl = NeuralNetworkImpl(networkDescription, lossFunctionType);
}

void NeuralNetwork::fit(NetworkTrainData trainData, int epochs, int batchSize) {
      NeuralNetworkFit net_fit(trainData, epochs, batchSize, &net_impl);
      net_fit.fit();
}

std::vector<double> NeuralNetwork::predict(InputNetworkData input) {
      if (input.size() != net_impl.getInputSize())
            return std::vector<double>{-1};
      return net_impl._predict(input);
}
