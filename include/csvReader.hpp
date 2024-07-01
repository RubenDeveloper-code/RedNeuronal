#ifndef __CSV_READER_HPP__
#define __CSV_READER_HPP__

#include "Data.hpp"
#include <algorithm>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
class CSVReader {
    public:
      CSVReader(std::string path);
      // por implementar el bache de comprobacion
      NetworkTrainData toNetworkTrainData(std::vector<std::string> inputColumns,
                                          std::vector<std::string> ouputColumns,
                                          int percentBatch);
      ~CSVReader() { csvFile.close(); }

    private:
      int getColumnIndex(std::string name);
      void getFieldsName();
      std::ifstream csvFile;
      std::string readLine();
      std::vector<std::string> fieldNames;
      std::vector<double> readNextRow();
      Data toData(std::vector<double> row,
                  std::vector<std::string> inputColumns,
                  std::vector<std::string> ouputColumns);
};
#endif
