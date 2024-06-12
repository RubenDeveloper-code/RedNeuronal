#ifndef __NEURALNETWORK_IMPL_H__
#define __NEURALNETWORK_IMPL_H__

#include "Data.hpp"
#include "Layer.hpp"
#include "Neuron.hpp"
#include <vector>

class NeuralNetworkImpl {
    public:
      using NetworkDescription = std::vector<LayerDescription>;
      using Network = std::vector<Layer>;
      NeuralNetworkImpl(NetworkDescription);
      std::vector<double> _predict(InputNetworkData input);
      OutputNetworkData generateOutput();
      void recalculateWeights();

      static Layer *input, *output;
      inline int getInputSize() { return input->nNeurons; }
      inline int getOutputSize() { return output->nNeurons; }
      ~NeuralNetworkImpl() {}

    private:
      NetworkDescription networkDescription;
      Network network;
      bool networkIsValid();
      void buildNetwork();
      void alias_IO_layers();
      void createNetwork();
      void connectNetwork();
      void connect(Layer &layer, Neuron::Neurons &neurons);
};

#endif
