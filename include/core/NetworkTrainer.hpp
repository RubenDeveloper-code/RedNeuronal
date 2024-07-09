#ifndef __NETWORK_TRAINER_HPP__
#define __NETWORK_TRAINER_HPP__

#include "../algorithms/AlphaAlgoritms.hpp"
#include "../designs/AlgorithmsSpects.hpp"
#include "../designs/TrainSpects.hpp"
#include "Network.hpp"
#include "NetworkOperator.hpp"
#include "SetterData.hpp"
#include <vector>
class NetworkTrainer {
    public:
      void fit(Network &network, SharedResources &shared_resources,
               TrainSpects trainSpects, AlgorithmsSpects &algorithms_spects,
               NetworkOperator &network_operator,
               std::shared_ptr<LossFuctions::LossFunction> loss_function);

    private:
      SetterData setter_data;
      std::unique_ptr<AlphaAlgorithms::AlphaAlgorithm> alpha_algorithm;
      std::vector<PairOutputs>
      forwardPropagation(Network &network, NetworkOperator &network_operator,
                         double mini_batch);
      double
      iteration(Network &network, TrainSpects &train_spects,
                NetworkOperator &network_operator,
                std::shared_ptr<LossFuctions::LossFunction> loss_function);
      PairOutputs
      individualDataForewardPropagation(Network &network,
                                        NetworkOperator &network_operator);

      void backpropagation(Network &network, NetworkOperator &network_operator,
                           std::vector<PairOutputs> mini_batch_outputs);
      bool checkIfReachUmbrall(double loss, double umbral, double epoch);
      void initAlgorithms(AlgorithmsSpects &algorithms_spects);
      double
      computeLoss(std::vector<PairOutputs> pair_outputs,
                  std::shared_ptr<LossFuctions::LossFunction> loss_function);
      PairOutputs minibatchPerfomance(std::vector<PairOutputs> perfomance);
};

#endif
