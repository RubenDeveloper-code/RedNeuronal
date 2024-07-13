#ifndef __MODEL_HPP__
#define __MODEL_HPP__

#include "algorithms/AlphaAlgoritms.hpp"
#include "data/Data.hpp"
#include "designs/LayerDesign.hpp"
#include "designs/ModelDesign.hpp"
#include "designs/train/AlgorithmsSpects.hpp"
#include "designs/train/AlphaAlgoritmsSpects.hpp"
#include "designs/train/EarlyStopSpects.hpp"
#include "designs/train/TrainSpects.hpp"
#include "network/container/NeuralNetwork.hpp"
#include <vector>
class Model {
    public:
      Model(ModelDesign::LossFuction loss_function);
      void addLayer(LayerDesign layer_design);
      void upAlphaAlgoritm(AlgorithmsSpects::AlphaModifier alfaModifier,
                           AlphaAlgorithmsSpects spects);
      void upEarlyStop(EarlyStopSpects earlystop_spects);
      void fit(TrainSpects train_spects);
      OutputNetworkData predict(std::vector<double> input);
      void loadCheckpoint(std::string path);

    private:
      ModelDesign model_design;
      AlgorithmsSpects algorithms_spects;
      NeuralNetwork neural_network;
};

#endif
