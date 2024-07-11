#ifndef __NETWORK_TRAINER_HPP__
#define __NETWORK_TRAINER_HPP__

#include "../../include/data/checkpoints.hpp"
#include "../../libs/ProgressBar.hpp"
#include "../algorithms/AlphaAlgoritms.hpp"
#include "../designs/AlgorithmsSpects.hpp"
#include "../designs/TrainSpects.hpp"
#include "Network.hpp"
#include "NetworkOperator.hpp"
#include "SetterData.hpp"
#include <vector>
class NetworkTrainer {
    public:
      enum class Status { FITTING, DONE, RELOAD } status;
      Status fit(Network &network, SharedResources &shared_resources,
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
      double checkValidationLoss(
          Network &network, NetworkOperator &network_operator,
          DataSet &validation_data,
          std::shared_ptr<LossFuctions::LossFunction> loss_function);
      std::vector<PairOutputs>
      validationOutputs(Network &network, NetworkOperator &network_operator,
                        DataSet &validation_data);
      bool checkIfReachUmbrall(double loss, double umbral, double epoch);
      void initAlgorithms(AlgorithmsSpects &algorithms_spects);
      double
      computeLoss(std::vector<PairOutputs> pair_outputs,
                  std::shared_ptr<LossFuctions::LossFunction> loss_function);
      PairOutputs minibatchPerfomance(std::vector<PairOutputs> perfomance);
      void adminCheckpoint(Network &network, NetworkOperator &network_operator,
                           TrainSpects &train_spects, int a_epoch,
                           double validation_loss);
      bool checkIfCheckpoint(int a_epoch, int frec);
      void saveCheckpoint(Network &network, NetworkOperator &network_operator,
                          std::string dest_forlder,
                          Checkpoint::TYPE_CKPT type_ckpt);
      void updateBar(ProgressBar &progress_bar, int epochs_p, int epochs,
                     double loss, double validation_loss);
      void displayElapsedTime(ProgressBar &bar, double segs);
      void earlyStop(bool restart);
};
#endif
