#ifndef __NEURALNETWORK_HPP__
#define __NEURALNETWORK_HPP__

#include "../../algorithms/AlphaAlgoritms.hpp"
#include "../../designs/ModelDesign.hpp"
#include "../../designs/train/AlgorithmsSpects.hpp"
#include "../../designs/train/TrainSpects.hpp"
#include "../Network.hpp"
#include "../operator/networkOperator.hpp"
#include "../resources/SharedResources.hpp"
#include <memory>
class NeuralNetwork {
    public:
      Network network;
      NeuralNetwork(ModelDesign &modelDesign);
      void construct();
      void fit(TrainSpects &train_spects, AlgorithmsSpects &algorithms_spects);
      OutputNetworkData predict(InputNetworkData input);
      void loadCheckpoint(std::string path);
      void saveCheckpoint(std::string dest_forlder);

    private:
      NetworkOperator network_operator;
      ModelDesign &model_design;
      SharedResources shared_resources;
      std::shared_ptr<LossFuctions::LossFunction> loss_function;
      bool builded;
};

#endif
