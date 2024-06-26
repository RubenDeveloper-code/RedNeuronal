#ifndef __NEURALNETWORK_IMPL_H__
#define __NEURALNETWORK_IMPL_H__

#include "Data.hpp"
#include "Layer.hpp"
#include "LossFuctions.hpp"
#include "NetworkAlgoritms.hpp"
#include "NetworkGlobalResources.hpp"
#include "Neuron.hpp"
#include <memory>
#include <vector>

class NeuralNetworkImpl {
    public:
      using NetworkDescription = std::vector<LayerDescription>;
      using Network = std::vector<Layer>;
      NeuralNetworkImpl(NetworkDescription, LossFuctions::TYPE,
                        GlobalResourses &&);
      std::vector<double> _predict(InputNetworkData input);
      OutputNetworkData generateOutput();
      void recalculateWeights(std::vector<OutputNetworkData> acts,
                              std::vector<OutputNetworkData> targets);

      Network network;
      NetworkAlgorithms::AlgorithmsAlpha netAlgorithmsAlpha;
      GlobalResourses GLOBAL_RESOURSES;
      Layer *getInputLayer() { return input; }
      Layer *getOutputLayer() { return output; }
      inline int getInputSize() { return input->nNeurons; }
      inline int getOutputSize() { return output->nNeurons; }
      std::shared_ptr<LossFuctions::LossFunction> lossFunction;
      void initAlphaAlgorithms();
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
