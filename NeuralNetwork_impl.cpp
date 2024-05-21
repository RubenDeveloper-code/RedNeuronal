#include "NeuralNetwork_impl.hpp"
#include "Neuron.hpp"
#include <algorithm>
#include <iostream>
#include <iterator>
#include <vector>

NeuralNetwork_impl::NeuralNetwork_impl(NetworkDimentions netDims,
                                       Neuron::Activation act)
    : activation(act), networkDimentions(netDims) {
      buildNetwork();
}

void NeuralNetwork_impl::buildNetwork() {
      createLayers();
      connectNetwork();
}

void NeuralNetwork_impl::createLayers() {
      inLayer = Layer{Neuron::TYPE::INPUT, networkDimentions.INPUT_NEURONS,
                      activation};
      outLayer = Layer{Neuron::TYPE::OUTPUT, networkDimentions.OUTPUT_NETWORKS,
                       activation};
      for (auto &wideLayerSize : networkDimentions.WIDE_LAYERS_NEURONS) {
            wideLayers.emplace_back(Neuron::TYPE::WIDE, wideLayerSize,
                                    [](double y) { return std::max(0.0, y); });
      }
}
void NeuralNetwork_impl::connectNetwork() {
      // Invariantes: No wide layers, no inputLayer, no Output layer
      connect(inLayer, wideLayers.begin()->neurons, Neuron::TYPE::WIDE);
      for (auto layer = wideLayers.begin();
           layer != std::prev(wideLayers.end()); layer++) {
            connect(*layer, std::next(layer)->neurons, Neuron::TYPE::WIDE);
      }

      connect(*std::prev(wideLayers.end()), outLayer.neurons,
              Neuron::TYPE::OUTPUT);
}

void NeuralNetwork_impl::connect(Layer &layer, Neuron::Neurons &neurons,
                                 Neuron::TYPE type) {
      for (auto &layerNeuron : layer.neurons) {
            layerNeuron.makeConnections(neurons, layer.neurons.size(), type);
      }
}

std::vector<double> NeuralNetwork_impl::GenerateOutput(NetworkData input,
                                                       NetworkData output) {
      initEpoch(input, output);
      for (auto &layer : wideLayers) {
            for (auto &neuron : layer.neurons) {
                  neuron.calculateValue();
            }
      }
      // por aca ver la concordancia de los datos
      std::vector<double> netOut;
      for (auto &out : outLayer.neurons) {
            netOut.push_back(out.calculateValue());
      }
      return netOut;
}
// aca andamos
void NeuralNetwork_impl::RecalculateWeights() {
      for (auto &out : outLayer.neurons) {
            out.fixInputWeights();
      }

      for (auto layer = wideLayers.rbegin(); wideLayers.rend() != layer;
           layer++) {
            for (auto &neuron : layer->neurons) {
                  neuron.fixInputWeights();
            }
      }
}

void NeuralNetwork_impl::initEpoch(NetworkData inputBatch,
                                   NetworkData outputBatch) {
      for (long unsigned int i = 0; i < inputBatch.size(); i++) {
            inLayer.neurons[i].setValue(inputBatch[i]);
      }
      for (long unsigned int i = 0; i < outputBatch.size(); i++) {
            outLayer.neurons[i].setValue(outputBatch[i]);
      }
}
