#include "../../include/algorithms/LossFuctions.hpp"

double LossFuctions::LossFunction::summation(OutputNetworkData activations,
                                             OutputNetworkData targets,
                                             double (*ecuation)(double,
                                                                double)) {
      const int OUTPUT_SIZE = activations.size();
      double summation{};
      double loss{};
      for (int itSingle = 0; itSingle < OUTPUT_SIZE; itSingle++) {
            summation += ecuation(targets[itSingle], activations[itSingle]);
      }
      loss = summation / (OUTPUT_SIZE);
      return loss;
}

std::shared_ptr<LossFuctions::LossFunction>
LossFuctions::newInstance(LossFuctions::TYPE type) {
      switch (type) {
      case LossFuctions::TYPE::MEAN_SQUARED_ERROR:
            return std::make_shared<LossFuctions::MeanSquaredError>();
      case LossFuctions::TYPE::BINARY_CROSS_ENTROPY:
            return std::make_shared<LossFuctions::BinaryCrossEntropy>();
      }
      return std::make_shared<LossFuctions::MeanSquaredError>();
}
