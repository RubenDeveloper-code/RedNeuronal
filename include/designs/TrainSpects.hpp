#ifndef __TRAIN_SPECTS_HPP__
#define __TRAIN_SPECTS_HPP__

#include "../types/TrainingDataSet.hpp"
struct TrainSpects {
      TrainSpects(TrainingDataSet &&dataset, int epochs, int mini_batch,
                  double umbral, double alpha, int checkpoint_frec,
                  std::string checkpoints_folder, unsigned patience,
                  bool earlyStop_restart)
          : dataset(std::move(dataset)), epochs(epochs), mini_batch(mini_batch),
            umbral(umbral), alpha(alpha), checkpoint_frec(checkpoint_frec),
            checkpoints_folder(checkpoints_folder), patience(patience),
            earlyStop_restart(earlyStop_restart){};
      TrainingDataSet dataset;
      const unsigned patience;
      const int epochs;
      const int mini_batch;
      const double umbral;
      const int checkpoint_frec;
      const std::string checkpoints_folder;
      const std::string tempcheckpoints_folder = "../temp";
      const bool earlyStop_restart;
      const double alpha;
};
#endif
