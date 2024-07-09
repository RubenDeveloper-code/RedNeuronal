#include "../../include/data/DataSetProcess.hpp"
#include "../../include/alerts/messages.hpp"
#include "../../include/data/csvReader.hpp"
#include <algorithm>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

DataSetProcess::DataSetProcess(std::string path) {
      CSVReader file(path);
      rows = file.toVector();
      names = file.getColumnNames();
}

DataSet DataSetProcess::getDataset(std::vector<std::string> inputColumns,
                                   std::vector<std::string> ouputColums) {
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
            for (auto &column_name : ouputColums) {
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
