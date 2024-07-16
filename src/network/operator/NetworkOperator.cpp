#include "../../../include/network/operator/networkOperator.hpp"
#include <algorithm>
#include <pstl/glue_algorithm_defs.h>
#include <random>
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

void NetworkOperator::applyDropout(Network &network) {
      std::vector<int> createMask(double, double);
      std::vector<int> mask;
      for (auto &layer : network) {
            if (layer.type != Layer::TYPE::OUTPUT && layer.p > 0.01) {
                  mask = createMask(layer.p, layer.neurons.size());
                  for (auto &neuron : layer) {
                        neuron.changeState(mask.back());
                        mask.pop_back();
                  }
            }
      }
}

void NetworkOperator::clearNetwork(Network &network) {
      for (auto &layer : network) {
            for (auto &neuron : layer) {
                  neuron.changeState(true);
            }
      }
}

std::vector<int> createMask(double p, double size) {
      std::vector<int> mask;
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<> dis(0.0, 1.0);
      const int targetDown = p * size;
      const int targetUp = (1.01 - p) * size;
      double downed{}, upted{};
      for (auto it = 0; it < size; it++) {
            double random_number = dis(gen);
            int imask = 0;
            if (random_number < p && targetDown != downed) {
                  downed++;
                  imask = 0;
            } else if (targetUp != upted) {
                  upted++;
                  imask = 1;
            }
            mask.push_back(imask);
      }

      return mask;
}
