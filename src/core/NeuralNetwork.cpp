#include "../../include/core/NeuralNetwork.hpp"
#include "../../include/alerts/messages.hpp"
#include "../../include/core/NetworkConstructor.hpp"
#include "../../include/core/NetworkTrainer.hpp"
#include "../../include/data/checkpoints.hpp"
#include <algorithm>
#include <memory>

NeuralNetwork::NeuralNetwork(ModelDesign &modelDesign)
    : model_design(modelDesign) {}

void NeuralNetwork::construct() {
      if (!builded) {
            shared_resources = std::move(SharedResources());
            model_design.checkIntegrity();
            NetworkConstructor network_constructor;
            loss_function = LossFuctions::newInstance(
                static_cast<LossFuctions::TYPE>(model_design.loss_function));
            network_constructor.construct(network, model_design,
                                          shared_resources, loss_function);
            builded = true;
      }
}
void NeuralNetwork::fit(TrainSpects &train_spects,
                        AlgorithmsSpects &algorithms_spects) {
      shared_resources.init(train_spects.alpha);
      NetworkTrainer network_trainer;
      NetworkTrainer::Status status = network_trainer.fit(
          network, shared_resources, train_spects, algorithms_spects,
          network_operator, loss_function);
      if (status == NetworkTrainer::Status::RELOAD) {
            loadCheckpoint(train_spects.tempcheckpoints_folder +
                           "/temp_best.ckpt");
            fit(train_spects, algorithms_spects);
      }
}

OutputNetworkData NeuralNetwork::predict(InputNetworkData input) {
      SetterData setter;
      setter.preparePrediction(network.input(), input);
      auto prediction = network_operator.computeNetworkOutput(network);
      return prediction;
}
void NeuralNetwork::loadCheckpoint(std::string path) {
      Checkpoint ckpt;
      auto network_parameters = ckpt.loadCheckpoint(path);
      network_operator.loadCheckpointParameters(network, network_parameters);
      Messages::Message({"checkpoint ", path, " loaded"});
}
