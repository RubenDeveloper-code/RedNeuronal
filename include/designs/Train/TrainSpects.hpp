#ifndef __TRAIN_SPECTS_HPP__
#define __TRAIN_SPECTS_HPP__

#include "../../types/TrainingDataSet.hpp"
#include "CheckpointSpects.hpp"
#include "EarlyStopSpects.hpp"
struct TrainSpects {
      TrainSpects(TrainingDataSet &&dataset, int epochs, int mini_batch,
                  double umbral, double alpha, int checkpoint_frec,
                  std::string checkpoints_folder, unsigned patience,
                  bool earlyStop_restart)
          : dataset(std::move(dataset)), epochs(epochs), mini_batch(mini_batch),
            umbral(umbral), alpha(alpha),
            checkpoints_spects(checkpoint_frec, checkpoints_folder),
            earlystop_spects(patience, earlyStop_restart){};
      TrainingDataSet dataset;
      CheckpointSpects checkpoints_spects;
      EarlyStopSpects earlystop_spects;
      const int epochs;
      const int mini_batch;
      const double umbral;
      const double alpha;
};
#endif
