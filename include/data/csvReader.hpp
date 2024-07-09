#ifndef __CSV_READER_HPP__
#define __CSV_READER_HPP__

#include <fstream>
#include <string>
#include <vector>
using CSVector = std::vector<std::vector<double>>;
class CSVReader {
    public:
      CSVReader(std::string path);
      CSVector toVector();
      std::vector<std::string> &getColumnNames();
      ~CSVReader() { csvFile.close(); }

    private:
      int getColumnIndex(std::string name);
      void readFieldsName();
      std::ifstream csvFile;
      std::string readLine();
      std::vector<std::string> fieldNames;
      std::vector<double> readNextRow();
};
#endif
