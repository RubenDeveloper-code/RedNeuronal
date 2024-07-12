#ifndef __NETWORK_ITERATION_HPP__
#define __NETWORK_ITERATION_HPP__
#include "../../designs/train/AlgorithmsSpects.hpp"
#include "../../designs/train/TrainSpects.hpp"
#include "../Network.hpp"
#include "../operator/SetterData.hpp"
#include "../operator/networkOperator.hpp"
#include "ModelLoss.hpp"
#include <memory>
class Iteration {
    public:
      Iteration(Network &network, SetterData &&setter_data,
                ModelLoss &model_loss, unsigned minibatch);
      double iterate();
      double validationForwardPropagation(DataSet &validation_data);

    private:
      Network &network;
      SetterData setter_data;
      ModelLoss &model_loss;
      int minibatch_size;
      NetworkOperator network_operator;
      std::vector<PairOutputs> forwardPropagation();
      PairOutputs individualDataForewardPropagation();
      void backpropagation(std::vector<PairOutputs> mini_batch_outputs);
};

#endif
