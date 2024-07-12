#ifndef __NETWORK_TRAINER_HPP__
#define __NETWORK_TRAINER_HPP__

#include "../../../include/core/NetworkTrainer/Checks.hpp"
#include "../../../include/core/NetworkTrainer/Iteration.hpp"
#include "../../../include/core/NetworkTrainer/ModelCheckpoint.hpp"
#include "../../../include/core/NetworkTrainer/ModelLoss.hpp"
#include "../../../include/core/NetworkTrainer/TrainerUI.hpp"
#include "../../algorithms/AlphaAlgoritms.hpp"
#include "../../designs/AlgorithmsSpects.hpp"
#include "../../designs/Train/TrainSpects.hpp"
#include "../Network.hpp"
#include "../NetworkOperator.hpp"
#include "../SetterData.hpp"
#include <memory>
#include <vector>
class Trainer {
    public:
      enum class Status { FITTING, DONE, RELOAD } status;
      Trainer(Network &network, SharedResources &shared_resources,
              TrainSpects &train_spects, AlgorithmsSpects &algorithms_spects,
              std::shared_ptr<LossFuctions::LossFunction> loss_funcion);
      Status fit();

    private:
      Network &network;
      SetterData setter_data;
      NetworkOperator network_operator;
      SharedResources &shared_resources;
      ModelLoss model_loss;
      Iteration iteration;
      ModelCheckpoint model_ckpt;
      Checks checks;
      TrainerUI trainer_ui;

      TrainSpects &trainer_spects;
      std::unique_ptr<AlphaAlgorithms::AlphaAlgorithm> alpha_algorithm;
      std::shared_ptr<LossFuctions::LossFunction> loss_funcion;

      void initAlgorithms(AlgorithmsSpects &algorithms_spects);
};
#endif
