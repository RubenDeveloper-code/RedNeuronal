#include "../../include/data/DataSetProcess.hpp"
#include "../../include/alerts/messages.hpp"
#include "../../include/data/csvReader.hpp"
#include <algorithm>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

int GET_AMOUNT(double percent, double max) {
      double perc = (percent / 100.0) * max;
      return perc;
}

DataSetProcess::DataSetProcess(std::string path) {
      CSVReader file(path);
      rows = file.toVector();
      names = file.getColumnNames();
}

TrainingDataSet DataSetProcess::getTrainingDataSet(
    std::vector<std::string> inputColumns, std::vector<std::string> ouputColums,
    int perc_training, int perc_validation, int perc_test) {
      auto entireDataset = getFullDataset(
          inputColumns, ouputColums, perc_training, perc_validation, perc_test);
      return divideDataSet(entireDataset, perc_training, perc_validation,
                           perc_test);
}

DataSet DataSetProcess::getFullDataset(std::vector<std::string> inputColumns,
                                       std::vector<std::string> ouputColumns,
                                       int perc_training, int perc_validation,
                                       int perc_test) {
      DataSet dataset;
      dataset.column_names = names;
      int getColumnIndex(std::vector<std::string>, std::string);
      const int n_rows = rows.size();
      Messages::Message({"processing dataset"});
      for (auto &row : rows) {
            std::vector<double> input_elements;
            for (auto &column_name : inputColumns) {
                  double element =
                      row[getColumnIndex(dataset.column_names, column_name)];
                  input_elements.push_back(element);
            }
            std::vector<double> output_elements;
            for (auto &column_name : ouputColumns) {
                  double element =
                      row[getColumnIndex(dataset.column_names, column_name)];
                  output_elements.push_back(element);
            }
            dataset.set.emplace_back(Data{input_elements, output_elements});
      }
      if (normalization != nullptr) {
            normalization->normalizeData(dataset);
            Messages::Message({"Applied normalization"});
      }
      return dataset;
}

// Pa que
void DataSetProcess::dropColumn(std::string name) {
      int getColumnIndex(std::vector<std::string>, std::string);
      int index = getColumnIndex(names, name);
      for (auto &row : rows) {
            row.erase(row.begin() + index);
      }
}

void DataSetProcess::applyNormalization(Normalizations::TYPE type) {
      normalization = Normalizations::newInstance(type);
}
int getColumnIndex(std::vector<std::string> names, std::string name) {
      auto pred = [&name](std::string x) { return name == x; };
      auto it = std::find_if(names.begin(), names.end(), pred);
      int index = std::distance(names.begin(), it);
      return index;
}

TrainingDataSet DataSetProcess::divideDataSet(DataSet &dataset,
                                              int perc_training,
                                              int perc_validation,
                                              int perc_test) {
      const float size_dataset = dataset.size();
      DataSet training_dataset(GET_AMOUNT(perc_training, size_dataset));
      DataSet validation_dataset(GET_AMOUNT(perc_validation, size_dataset));
      DataSet test_dataset(GET_AMOUNT(perc_test, size_dataset));
      std::copy(dataset.begin(), dataset.begin() + training_dataset.size(),
                training_dataset.begin());
      std::copy(dataset.begin() + training_dataset.size() - 1,
                dataset.begin() + training_dataset.size() - 1 +
                    validation_dataset.size(),
                validation_dataset.begin());
      std::copy(dataset.begin() + training_dataset.size() - 1 +
                    validation_dataset.size() - 1,
                dataset.begin() + training_dataset.size() - 1 +
                    validation_dataset.size() - 1 + test_dataset.size(),
                test_dataset.begin());

      training_dataset.column_names = dataset.column_names;
      test_dataset.column_names = dataset.column_names;
      validation_dataset.column_names = dataset.column_names;

      return TrainingDataSet{std::move(training_dataset),
                             std::move(validation_dataset),
                             std::move(test_dataset)};
}

/* QUEDA POR SI SE OCUPA
std::vector<std::vector<std::reference_wrapper<double>>>
DataSetProcess::getColumnsRefs(CSVector preliminarData) {
      std::vector<std::vector<std::reference_wrapper<double>>> columns_refs;
      std::vector<std::reference_wrapper<double>> column_refs;
      const int N_COLUMNS = preliminarData[0].size();
      const int AMOUNT_DATA = preliminarData.size();
      for (auto it_element = 0; it_element < N_COLUMNS; it_element++) {
            for (auto it = 0; it < AMOUNT_DATA; it++) {
                  column_refs.push_back(preliminarData[it][it_element]);
            }
            columns_refs.push_back(column_refs);
      }
      return columns_refs;
}
*/
