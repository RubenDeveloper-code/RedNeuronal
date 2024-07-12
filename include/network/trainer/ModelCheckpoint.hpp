#ifndef __MODEL_CKPT_HPP__
#define __MODEL_CKPT_HPP__

#include "../../designs/train/CheckpointSpects.hpp"
#include "../Network.hpp"
#include "../operator/checkpoints.hpp"
#include "../operator/networkOperator.hpp"

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
