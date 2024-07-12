#include "../../../include/network/operator/networkOperator.hpp"
#include <algorithm>
#include <pstl/glue_algorithm_defs.h>
#include <ranges>
#include <vector>
OutputNetworkData NetworkOperator::computeNetworkOutput(Network &network) {
      OutputNetworkData output;
      for (auto &layer : network) {
            if (layer.type == Layer::TYPE::INPUT)
                  continue;
            std::for_each(layer.begin(), layer.end(),
                          [&output](Neuron &neuron) {
                                auto out = neuron.computeActivation();
                                if (neuron.type == Neuron::TYPE::OUTPUT)
                                      output.push_back(out);
                          });
      }
      return output;
}
void NetworkOperator::recalculateNetworkParameters(
    Network &network, std::vector<PairOutputs> pair_outputs) {
      auto responsabilities_neurons =
          distributeResponsibilities(pair_outputs, network.ouputSize());
      for (auto &layer : std::ranges::reverse_view(network)) {
            int index = 0;
            std::for_each(
                layer.begin(), layer.end(),
                [&responsabilities_neurons, &index](Neuron &neuron) {
                      if (neuron.type == Neuron::TYPE::OUTPUT) {
                            neuron.recomputeParameters(
                                responsabilities_neurons[index].computed,
                                responsabilities_neurons[index].desired);
                            index++;
                      } else {
                            neuron.recomputeParameters();
                      }
                });
      }
}
std::vector<PairOutputs> NetworkOperator::distributeResponsibilities(
    std::vector<PairOutputs> pair_outputs, int size_output) {
      std::vector<PairOutputs> responsabilities;
      for (auto it_element = 0; it_element < size_output; it_element++) {
            PairOutputs neuron_responsability;
            for (auto it_iteration = 0; it_iteration < pair_outputs.size();
                 it_iteration++) {
                  neuron_responsability.computed.push_back(
                      pair_outputs[it_iteration].computed[it_element]);
                  neuron_responsability.desired.push_back(
                      pair_outputs[it_iteration].desired[it_element]);
            }
            responsabilities.push_back(neuron_responsability);
      }
      return responsabilities;
}

std::vector<Parameters>
NetworkOperator::getNetworkParameters(Network &network) {
      std::vector<Parameters> network_params;
      for (auto &layer : network) {
            if (layer.type != Layer::TYPE::INPUT) {
                  for (auto &neuron : layer) {
                        auto params = neuron.getParameters();
                        network_params.push_back(params);
                  }
            }
      }
      return network_params;
}

void NetworkOperator::loadCheckpointParameters(
    Network &network, std::vector<Parameters> network_params) {
      std::vector<Parameters>::iterator neuron_parameters =
          network_params.begin();
      for (auto &layer : network) {
            if (layer.type != Layer::TYPE::INPUT) {
                  for (auto &neuron : layer) {
                        neuron.loadParameters(*neuron_parameters);
                        neuron_parameters++;
                  }
            }
      }
}
