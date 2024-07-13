#include "../../../include/network/trainer/ModelCheckpoint.hpp"
ModelCheckpoint::ModelCheckpoint(Network &network,
                                 CheckpointSpects &checkpoints_spects,
                                 TrainSpects &train_spects,
                                 AlgorithmsSpects &algorithms_spects)
    : network(network), checkpoints_spects(checkpoints_spects),
      network_operator(), trainer_spects(train_spects),
      algorithms_spects(algorithms_spects) {}

void ModelCheckpoint::adminCheckpoint(int a_epoch, double validation_loss) {
      static double best_validation_loss = validation_loss;
      if (validation_loss < best_validation_loss) {
            best_validation_loss = validation_loss;
            if (a_epoch % checkpoints_spects.checkpoint_frec == 0) {
                  saveCheckpoint(Checkpoint::TYPE_CKPT::SAVE);
            }
            saveCheckpoint(Checkpoint::TYPE_CKPT::TEMP);
      }
}

void ModelCheckpoint::saveCheckpoint(Checkpoint::TYPE_CKPT type_ckpt) {
      std::string dest_folder;
      if (type_ckpt == Checkpoint::TYPE_CKPT::SAVE)
            dest_folder = checkpoints_spects.checkpoints_folder;
      else if (type_ckpt == Checkpoint::TYPE_CKPT::TEMP)
            dest_folder = checkpoints_spects.tempcheckpoints_folder;

      Checkpoint ckpt;
      auto network_params = network_operator.getNetworkParameters(network);
      ckpt.createCheckpoint(std::move(network_params), dest_folder,
                            trainer_spects, algorithms_spects, type_ckpt);
}
