#include "NeuralNetworkImpl.hpp"
#include "Data.hpp"
#include "Layer.hpp"
#include "NeuralNetworkSetterData.hpp"
#include "Neuron.hpp"
#include <algorithm>
#include <iterator>
#include <vector>

Layer *NeuralNetworkImpl::input = nullptr;
Layer *NeuralNetworkImpl::output = nullptr;

NeuralNetworkImpl::NeuralNetwork_impl(NetworkDescription networkDescription)
    : networkDescription(networkDescription) {
      buildNetwork();
}

void NeuralNetworkImpl::buildNetwork() {
      if (networkIsValid()) {
            createNetwork();
            connectNetwork();
      } else {
            return;
      }
}

bool NeuralNetworkImpl::networkIsValid() {
      bool valid = false;
      int hasInput{}, hasOutput{};
      for (auto layer : networkDescription) {
            if (layer.type == Neuron::TYPE::INPUT)
                  hasInput++;
            if (layer.type == Neuron::TYPE::OUTPUT)
                  hasOutput++;
      }
      if (hasInput == 1 && hasOutput == 1)
            valid = true;
      if (hasInput > 1)
            valid = false;
      if (hasOutput > 1)
            valid = false;
      return valid;
}

void NeuralNetworkImpl::createNetwork() {
      Network buildingNetwork;
      for (auto layerInfo : networkDescription) {
            buildingNetwork.emplace_back(
                Layer(layerInfo.type, layerInfo.activation, layerInfo.algorithm,
                      layerInfo.nNeurons));
      }
      std::sort(buildingNetwork.begin(), buildingNetwork.end(),
                [](const Layer &a, const Layer &b) { return a.type < b.type; });
      network = std::move(buildingNetwork);
      alias_IO_layers();
}

void NeuralNetworkImpl::alias_IO_layers() {
      for (auto &layer : network) {
            if (layer.type == Neuron::TYPE::INPUT)
                  NeuralNetworkImpl::input = &layer;
            if (layer.type == Neuron::TYPE::OUTPUT)
                  NeuralNetworkImpl::output = &layer;
      }
}
void NeuralNetworkImpl::connectNetwork() {
      for (auto layer = network.begin(); layer != std::prev(network.end());
           layer++) {
            if (layer->type != Neuron::TYPE::OUTPUT)
                  connect(*layer, std::next(layer)->neurons);
      }
}

void NeuralNetworkImpl::connect(Layer &layer, Neuron::Neurons &neurons) {
      for (auto &layerNeuron : layer.neurons) {
            layerNeuron.makeConnections(neurons, layer.neurons.size());
      }
}

std::vector<double> NeuralNetworkImpl::_predict(InputNetworkData inputData) {
      SetterData predictSetter{};
      predictSetter.preparePrediction(input, inputData);
      return generateOutput();
}

OutputNetworkData NeuralNetworkImpl::generateOutput() {
      std::vector<double> netOut;
      for (auto &layer : network) {
            if (layer.type != Neuron::TYPE::INPUT) {
                  for (auto &neuron : layer.neurons) {
                        auto out = neuron.calculateValue();
                        if (layer.type == Neuron::TYPE::OUTPUT)
                              netOut.push_back(out);
                  }
            }
      }
      return netOut;
}
void NeuralNetworkImpl::recalculateWeights() {
      for (auto layer = network.rbegin(); std::prev(network.rend()) != layer;
           layer++) {
            for (auto &neuron : layer->neurons) {
                  neuron.fixInputWeights();
            }
      }
}
