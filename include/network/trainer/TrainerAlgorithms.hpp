#ifndef __ALGORITHMS_TRAIN_HPP__
#define __ALGORITHMS_TRAIN_HPP__

#include "../../../include/designs/train/AlgorithmsSpects.hpp"
#include <memory>

class TrainerAlgorithms {
    public:
      TrainerAlgorithms(AlgorithmsSpects &algorithms_spects,
                        SharedResources &shared_resources);
      void runAlphaAlgorithm();
      bool runEarlyStopAdmin(double validation_loss);
      void restart();

    private:
      AlgorithmsSpects &algorithms_spects;
      SharedResources &shared_resources;
      std::shared_ptr<AlphaAlgorithms::AlphaAlgorithm> alpha_algorithm;
      void initAlgorithms(AlgorithmsSpects &algorithms_spects);
};

#endif
