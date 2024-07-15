#ifndef __MODEL_CKPT_HPP__
#define __MODEL_CKPT_HPP__

#include "../../designs/train/AlgorithmsSpects.hpp"
#include "../../designs/train/CheckpointSpects.hpp"
#include "../../designs/train/TrainSpects.hpp"
#include "../Network.hpp"
#include "../operator/checkpoints.hpp"
#include "../operator/networkOperator.hpp"

class ModelCheckpoint {
    public:
      ModelCheckpoint(Network &network, CheckpointSpects &checkpoints_spects,
                      TrainSpects &train_spects,
                      AlgorithmsSpects &algorithms_spects);
      void adminCheckpoint(int a_epoch, double validation_loss);

    private:
      Network &network;
      TrainSpects &trainer_spects;
      AlgorithmsSpects &algorithms_spects;
      CheckpointSpects &checkpoints_spects;
      NetworkOperator network_operator;
      void saveCheckpoint(Checkpoint::TYPE_CKPT type_ckpt, int actual_epoch);
};

#endif
