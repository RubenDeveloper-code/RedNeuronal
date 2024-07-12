#ifndef __NETWORK_ITERATION_HPP__
#define __NETWORK_ITERATION_HPP__
#include "../../algorithms/AlphaAlgoritms.hpp"
#include "../../designs/AlgorithmsSpects.hpp"
#include "../../designs/Train/TrainSpects.hpp"
#include "../Network.hpp"
#include "../NetworkOperator.hpp"
#include "../SetterData.hpp"
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
