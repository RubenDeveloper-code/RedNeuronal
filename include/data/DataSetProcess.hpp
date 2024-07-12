#ifndef __TRAIN_DATA_HPP__
#define __TRAIN_DATA_HPP__

#include "../types/data/TrainingDataSet.hpp"
#include "Data.hpp"
#include "DataSet.hpp"
#include "Normalizers.hpp"
#include "csvReader.hpp"
#include <string>
#include <vector>

class DataSetProcess {
    public:
      DataSetProcess(std::string path);
      void dropColumn(std::string to_drop);
      TrainingDataSet getTrainingDataSet(std::vector<std::string> inputs,
                                         std::vector<std::string> ouputs,
                                         int perc_training, int perc_validation,
                                         int perc_test);
      void applyNormalization(Normalizations::TYPE type);

    private:
      std::vector<std::vector<std::reference_wrapper<double>>>
      getColumnsRefs(CSVector preliminarData);
      std::vector<std::vector<double>> rows;
      std::vector<std::string> names;
      std::unique_ptr<Normalizations::Normalization> normalization;
      DataSet getFullDataset(std::vector<std::string> inputs,
                             std::vector<std::string> ouputs, int perc_training,
                             int perc_validation, int perc_test);
      TrainingDataSet divideDataSet(DataSet &dataset, int perc_training,
                                    int perc_validation, int perc_test);
};

#endif
