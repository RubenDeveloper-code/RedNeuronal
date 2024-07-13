#include "../include/Model.hpp"
#include <memory>
Model::Model(ModelDesign::LossFuction loss_function)
    : model_design(), neural_network(model_design) {
      model_design.loss_function = loss_function;
}

void Model::addLayer(LayerDesign layer_design) {
      layer_design.loss_function =
          static_cast<LayerDesign::LossFuctions>(model_design.loss_function);
      model_design.design.emplace_back(layer_design);
}

void Model::upAlphaAlgoritm(AlgorithmsSpects::AlphaModifier alfaModifier,
                            AlphaAlgorithmsSpects spects) {
      algorithms_spects.alphaModifier = alfaModifier;
      algorithms_spects.args_alpha_modifier =
          std::make_unique<AlphaAlgorithmsSpects>(spects);
}

void Model::upEarlyStop(EarlyStopSpects earlystop_spects) {
      algorithms_spects.earlystop_spects = earlystop_spects;
}

void Model::fit(TrainSpects train_spects) {
      neural_network.construct();
      neural_network.fit(train_spects, algorithms_spects);
}

OutputNetworkData Model::predict(std::vector<double> input) {
      auto prediction = neural_network.predict(input);
      return prediction;
}

void Model::loadCheckpoint(std::string path) {
      neural_network.construct();
      neural_network.loadCheckpoint(path);
}
