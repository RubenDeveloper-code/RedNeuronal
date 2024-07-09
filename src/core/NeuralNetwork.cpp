#include "../../include/core/NeuralNetwork.hpp"
#include "../../include/core/NetworkConstructor.hpp"
#include "../../include/core/NetworkTrainer.hpp"
#include <algorithm>
#include <memory>

NeuralNetwork::NeuralNetwork(ModelDesign &modelDesign)
    : model_design(modelDesign) {}

void NeuralNetwork::construct() {
      shared_resources = std::move(SharedResources());
      model_design.checkIntegrity();
      NetworkConstructor network_constructor;
      loss_function = LossFuctions::newInstance(
          static_cast<LossFuctions::TYPE>(model_design.loss_function));
      network_constructor.construct(network, model_design, shared_resources,
                                    loss_function);
}
void NeuralNetwork::fit(TrainSpects &train_spects,
                        AlgorithmsSpects &algorithms_spects) {
      shared_resources.init(train_spects.alpha);
      NetworkTrainer network_trainer;
      network_trainer.fit(network, shared_resources, train_spects,
                          algorithms_spects, network_operator, loss_function);
}

OutputNetworkData NeuralNetwork::predict(InputNetworkData input) {
      SetterData setter;
      setter.preparePrediction(network.input(), input);
      auto prediction = network_operator.computeNetworkOutput(network);
      return prediction;
}
