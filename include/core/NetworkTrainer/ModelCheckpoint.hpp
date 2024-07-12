#ifndef __MODEL_CKPT_HPP__
#define __MODEL_CKPT_HPP__

#include "../../data/checkpoints.hpp"
#include "../../designs/Train/CheckpointSpects.hpp"
#include "../Network.hpp"
#include "../NetworkOperator.hpp"

class ModelCheckpoint {
    public:
      ModelCheckpoint(Network &network, CheckpointSpects &checkpoints_spects);
      void adminCheckpoint(int a_epoch, double validation_loss);

    private:
      Network &network;
      CheckpointSpects &checkpoints_spects;
      NetworkOperator network_operator;
      void saveCheckpoint(Checkpoint::TYPE_CKPT type_ckpt);
};

#endif
