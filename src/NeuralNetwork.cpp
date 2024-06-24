#include "../include/NeuralNetwork.hpp"
#include "../include/Data.hpp"
#include "../include/NeuralNetworkFit.hpp"
#include "../include/NeuralNetworkImpl.hpp"
#include <memory>
#include <vector>

NeuralNetwork::NeuralNetwork(NetworkDescription networkDescription,
                             LossFuctions::TYPE lossFunctionType)
    : epochs_ptr(std::make_shared<int>(0)),
      net_impl{networkDescription, lossFunctionType, epochs_ptr} {}

void NeuralNetwork::fit(NetworkTrainData trainData, int epochs, int batchSize) {
      NeuralNetworkFit net_fit(trainData, batchSize, epochs, epochs_ptr,
                               &net_impl);
      net_fit.fit();
}

std::vector<double> NeuralNetwork::predict(InputNetworkData input) {
      if (input.size() != net_impl.getInputSize())
            return std::vector<double>{-1};
      return net_impl._predict(input);
}
