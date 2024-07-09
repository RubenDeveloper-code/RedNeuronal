#include "../../include/data/csvReader.hpp"
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>

CSVReader::CSVReader(std::string path) : csvFile(path) {}

std::vector<std::vector<double>> CSVReader::toVector() {
      CSVector csv_vector;
      int n_lines = std::count(std::istreambuf_iterator<char>(csvFile),
                               std::istreambuf_iterator<char>(), '\n') -
                    1;
      void rollback_ifstream(std::ifstream &);
      rollback_ifstream(csvFile);
      readFieldsName();
      // error con 1 dato
      n_lines = 200;
      while (n_lines-- > 0) {
            std::vector<double> nextRowData = readNextRow();
            csv_vector.push_back(nextRowData);
      }
      return csv_vector;
}

std::vector<std::string> &CSVReader::getColumnNames() { return fieldNames; }

void CSVReader::readFieldsName() {
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

void rollback_ifstream(std::ifstream &ifs) {
      ifs.clear();
      ifs.seekg(0, std::ios::beg);
}
