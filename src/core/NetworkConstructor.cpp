#include "../../include/core/NetworkConstructor.hpp"
#include "../../include/alerts/messages.hpp"
#include "../../include/core/Layer.hpp"
#include <algorithm>
#include <iterator>
void NetworkConstructor::construct(
    Network &network, ModelDesign &model_design, SharedResources &shared_res,
    std::shared_ptr<LossFuctions::LossFunction> loss_function) {
      createLayers(network, model_design, shared_res, loss_function);
      connectLayers(network);
      Messages::Message({"built network"});
}
void NetworkConstructor::createLayers(
    Network &network, ModelDesign &model_design, SharedResources &shared_res,
    std::shared_ptr<LossFuctions::LossFunction> loss_function) {
      for (auto layer_design : model_design.design) {
            network.layers.emplace_back(
                Layer{layer_design, loss_function, shared_res});
      }
}
void NetworkConstructor::connectLayers(Network &network) {
      for (auto layer = network.layers.begin();
           layer != std::prev(network.layers.end()); layer++) {
            if (layer->type != Layer::TYPE::OUTPUT)
                  std::for_each(layer->neurons.begin(), layer->neurons.end(),
                                [&layer](Neuron &neuron) {
                                      neuron.makeConnections(
                                          std::next(layer)->neurons,
                                          layer->neurons.size());
                                });
      }
}
