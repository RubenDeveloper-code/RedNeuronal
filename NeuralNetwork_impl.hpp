#ifndef __NEURALNETWORK_IMPL_H__
#define __NEURALNETWORK_IMPL_H__

#include "Neuron.hpp"
#include <vector>
struct NetworkDimentions {
      int INPUT_NEURONS;
      std::vector<int> WIDE_LAYERS_NEURONS;
      int OUTPUT_NETWORKS;
};

struct Layer {
      Layer(){};
      Layer(Neuron::TYPE _type, int nNeurons, Neuron::Activation act)
          : type{_type} {
            while (nNeurons-- > 0) {
                  neurons.emplace_back(Neuron{act, _type});
            }
      }
      Neuron::TYPE type;
      Neuron::Neurons neurons;
};

class NeuralNetwork_impl {
    public:
      using NetworkData = std::vector<int>;
      NeuralNetwork_impl(NetworkDimentions netDims,
                         Neuron::Activation activation);
      std::vector<double> GenerateOutput(NetworkData inputBatch,
                                         NetworkData outputBatch);
      void RecalculateWeights();
      inline int getInputSize() { return networkDimentions.INPUT_NEURONS; }
      inline int getOutputSize() { return networkDimentions.OUTPUT_NETWORKS; }

    private:
      Neuron::Activation activation;
      NetworkDimentions networkDimentions;
      Layer inLayer, outLayer;
      std::vector<Layer> wideLayers;
      void buildNetwork();
      void createLayers();
      void connectNetwork();
      void connect(Layer &layer, Neuron::Neurons &neurons, Neuron::TYPE type);
      void initEpoch(NetworkData inputBatch, NetworkData outputBatch);
};

#endif
