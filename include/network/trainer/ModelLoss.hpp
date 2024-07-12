#ifndef __MODEL_LOSS_HPP__
#define __MODEL_LOSS_HPP__

#include "../../algorithms/LossFuctions.hpp"
#include "../../types/network/PairOutputs.hpp"
#include <memory>
class ModelLoss {
    public:
      ModelLoss(std::shared_ptr<LossFuctions::LossFunction> loss_funcion);
      double computeLoss(std::vector<PairOutputs> pair_outputs);

    private:
      std::shared_ptr<LossFuctions::LossFunction> loss_function;
      PairOutputs minibatchPerfomance(std::vector<PairOutputs> perfomance);
};
#endif
