#include "../../../include/network/body/Layer.hpp"
#include <memory>

Layer::Layer(LayerDesign layer_design,
             std::shared_ptr<LossFuctions::LossFunction> loss_function,
             SharedResources &shared_resources) {
      int n_neurons = layer_design.n_neurons;
      auto activation = Activations::newInstance(
          static_cast<Activations::TYPE>(layer_design.activation));
      auto optimizator = Optimizers::newInstance(
          static_cast<Optimizers::TYPE>(layer_design.optimizer),
          shared_resources);
      while (layer_design.n_neurons-- > 0) {
            neurons.emplace_back(
                Neuron{activation, optimizator, loss_function,
                       static_cast<Neuron::TYPE>(layer_design.type)});
      }
      type = static_cast<TYPE>(layer_design.type);
}
