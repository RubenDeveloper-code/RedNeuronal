#include "../include/csvReader.hpp"
#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

CSVReader::CSVReader(std::string path) : csvFile(path) {}

NetworkTrainData
CSVReader::toNetworkTrainData(std::vector<std::string> inputColumns,
                              std::vector<std::string> ouputColumns,
                              int percentBatch) {
      NetworkTrainData trainData;
      int nLines = std::count(std::istreambuf_iterator<char>(csvFile),
                              std::istreambuf_iterator<char>(), '\n') -
                   1;
      csvFile.clear();
      csvFile.seekg(0, std::ios::beg);
      getFieldsName();
      nLines = 10;
      while (nLines-- > 0) {
            std::vector<double> nextRowData = readNextRow();
            trainData.push_back(
                toData(nextRowData, inputColumns, ouputColumns));
      }
      return trainData;
}

Data CSVReader::toData(std::vector<double> row,
                       std::vector<std::string> inputColumns,
                       std::vector<std::string> ouputColumns) {
      Data data(inputColumns.size(), ouputColumns.size());
      for (auto itInput = 0; itInput < inputColumns.size(); itInput++) {
            data.input[itInput] = row[getColumnIndex(inputColumns[itInput])];
      }
      for (auto itOutput = 0; itOutput < ouputColumns.size(); itOutput++) {
            data.output[itOutput] = row[getColumnIndex(ouputColumns[itOutput])];
      }
      return data;
}

int CSVReader::getColumnIndex(std::string name) {
      auto it = std::find(fieldNames.begin(), fieldNames.end(), name);
      if (it != fieldNames.end()) {
            int index = std::distance(fieldNames.begin(), it);
            return index;
      }
      return -1;
}

void CSVReader::getFieldsName() {
      std::string temp_row;
      std::getline(csvFile, temp_row);
      char *row = new char[temp_row.size() + 1];
      strcpy(row, temp_row.c_str());
      const char *delim = ",";
      char *name = std::strtok(row, delim);
      while (name != nullptr) {
            fieldNames.push_back(name);
            name = std::strtok(NULL, delim);
      }
      delete[] row;
}

std::vector<double> CSVReader::readNextRow() {
      std::vector<double> dataRow;
      std::string temp_row;
      std::getline(csvFile, temp_row);
      char *row = new char[temp_row.size() + 1];
      strcpy(row, temp_row.c_str());
      const char *delim = ",";
      char *token = std::strtok(row, delim);
      while (token != nullptr) {
            if (std::find_if(token, token + strlen(token),
                             [](const int c) { return std::isdigit(c); })) {

                  dataRow.push_back(atof(token));
            } else {
                  exit(2);
            }
            token = std::strtok(NULL, delim);
      }
      delete[] row;
      return dataRow;
}
