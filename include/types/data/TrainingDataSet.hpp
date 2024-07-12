#ifndef __TRAINNINGDATASET_HPP__
#define __TRAINNINGDATASET_HPP__

#include "../../data/DataSet.hpp"

struct TrainingDataSet {
      TrainingDataSet(DataSet &&train, DataSet &&val, DataSet &&test)
          : training_dataset(std::move(train)),
            validation_dataset(std::move(val)), test_dataset(std::move(test)){};
      DataSet training_dataset;
      DataSet validation_dataset;
      DataSet test_dataset;
};

#endif
