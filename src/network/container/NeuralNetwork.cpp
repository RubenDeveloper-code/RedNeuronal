#include "../../../include/network/container/NeuralNetwork.hpp"
#include "../../../include/alerts/messages.hpp"
#include "../../../include/network/constructor/NetworkConstructor.hpp"
#include "../../../include/network/operator/checkpoints.hpp"
#include "../../../include/network/trainer/Trainer.hpp"
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
      Trainer network_trainer(network, shared_resources, train_spects,
                              algorithms_spects, loss_function);
      Trainer::Status status;
      do {
            status = network_trainer.fit();
            if (status == Trainer::Status::RELOAD) {
                  Messages::Message({"Rollback best checkpoint"});
                  loadCheckpoint(
                      train_spects.checkpoints_spects.tempcheckpoints_folder +
                      "/temp_best.ckpt");
                  network_trainer.restart();
            }
      } while (status == Trainer::Status::RELOAD);
      Messages::Message({"Model trained"});
}

OutputNetworkData NeuralNetwork::predict(InputNetworkData input) {
      SetterData setter;
      setter.preparePrediction(network.input(), input);
      auto prediction = network_operator.computeNetworkOutput(network);
      return prediction;
}
void NeuralNetwork::loadCheckpoint(std::string path) {
      Checkpoint ckpt;
      auto network_parameters =
          ckpt.loadCheckpoint(path, network.inputSize(), network.ouputSize(),
                              shared_resources.epochs_it);
      network_operator.loadCheckpointParameters(network, network_parameters);
      Messages::Message({"checkpoint ", path, " loaded"});
}
