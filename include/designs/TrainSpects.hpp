#ifndef __TRAIN_SPECTS_HPP__
#define __TRAIN_SPECTS_HPP__

#include "../data/DataSet.hpp"
struct TrainSpects {
      TrainSpects(DataSet dataset, int epochs, int mini_batch, double umbral,
                  int chech_percent_batch, double alpha)
          : dataset(dataset), epochs(epochs), mini_batch(mini_batch),
            umbral(umbral), check_percent_batch(chech_percent_batch),
            alpha(alpha){};
      DataSet dataset;
      const int epochs;
      const int mini_batch;
      const double umbral;
      const int check_percent_batch;
      const double alpha;
};
#endif
