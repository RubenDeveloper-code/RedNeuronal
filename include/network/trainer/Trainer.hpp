#ifndef __NETWORK_TRAINER_HPP__
#define __NETWORK_TRAINER_HPP__

#include "../../../include/network/trainer/Checks.hpp"
#include "../../../include/network/trainer/Iteration.hpp"
#include "../../../include/network/trainer/ModelCheckpoint.hpp"
#include "../../../include/network/trainer/ModelLoss.hpp"
#include "../../../include/network/trainer/TrainerUI.hpp"
#include "../../algorithms/AlphaAlgoritms.hpp"
#include "../../designs/train/AlgorithmsSpects.hpp"
#include "../../designs/train/TrainSpects.hpp"
#include "../Network.hpp"
#include "../operator/SetterData.hpp"
#include "../operator/networkOperator.hpp"
#include <memory>
#include <vector>
class Trainer {
    public:
      enum class Status { FITTING, DONE, RELOAD } status;
      Trainer(Network &network, SharedResources &shared_resources,
              TrainSpects &train_spects, AlgorithmsSpects &algorithms_spects,
              std::shared_ptr<LossFuctions::LossFunction> loss_funcion);
      Status fit();
      void restart();

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
