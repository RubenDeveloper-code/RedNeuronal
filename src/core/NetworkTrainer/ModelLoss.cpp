#include "../../../include/core/NetworkTrainer/ModelLoss.hpp"
#include "../../../include/core/SetterData.hpp"
#include <memory>

ModelLoss::ModelLoss(std::shared_ptr<LossFuctions::LossFunction> loss_function)
    : loss_function(loss_function){};

double ModelLoss::computeLoss(std::vector<PairOutputs> pair_outputs) {
      auto perfomance = minibatchPerfomance(pair_outputs);
      double loss =
          loss_function->function(perfomance.computed, perfomance.desired);
      return loss;
}

PairOutputs
ModelLoss::minibatchPerfomance(std::vector<PairOutputs> minibatchs_perfomance) {
      PairOutputs perfomance;
      for (auto mini_batch : minibatchs_perfomance) {
            std::for_each(mini_batch.computed.begin(),
                          mini_batch.computed.end(),
                          [&perfomance](double computed) {
                                perfomance.computed.push_back(computed);
                          });
            std::for_each(mini_batch.desired.begin(), mini_batch.desired.end(),
                          [&perfomance](double desired) {
                                perfomance.desired.push_back(desired);
                          });
      }
      return perfomance;
}
