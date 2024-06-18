#ifndef __NEURALNETWORK_IMPL_H__
#define __NEURALNETWORK_IMPL_H__

#include "Layer.hpp"
#include "LossFuctions.hpp"
#include "Neuron.hpp"
#include <memory>
#include <vector>

class NeuralNetworkImpl {
    public:
      using NetworkDescription = std::vector<LayerDescription>;
      using Network = std::vector<Layer>;
      NeuralNetworkImpl(){};
      NeuralNetworkImpl(NetworkDescription, LossFuctions::TYPE);
      std::vector<double> _predict(InputNetworkData input);
      OutputNetworkData generateOutput();
      void recalculateWeights();

      Network network;
      Layer *getInputLayer() { return input; }
      Layer *getOutputLayer() { return output; }
      inline int getInputSize() { return input->nNeurons; }
      inline int getOutputSize() { return output->nNeurons; }
      std::shared_ptr<LossFuctions::LossFunction> lossFunction;
      ~NeuralNetworkImpl() {}

    private:
      Layer *input, *output;
      NetworkDescription networkDescription;
      LossFuctions::TYPE lossFunctionType;
      bool networkIsValid();
      void buildNetwork();
      void alias_IO_layers();
      void createNetwork();
      void connectNetwork();
      void connect(Layer &layer, Neuron::Neurons &neurons);
};

#endif
