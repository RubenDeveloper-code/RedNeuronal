#ifndef __TRAINING_PARAMETER_HPP__
#define __TRAINING_PARAMETER_HPP__

#include "../designs/TrainSpects.hpp"
#include <memory>

struct TrainingParameters {
      TrainingParameters(TrainSpects &train_spects)
          : epochs(train_spects.epochs) {}
      const unsigned epochs;
      std::shared_ptr<int> actual_epoch;
}

#endif
