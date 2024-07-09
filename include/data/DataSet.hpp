#ifndef __DATASET_HPP__
#define __DATASET_HPP__

#include "Data.hpp"
#include <string>
#include <utility>
#include <vector>
struct DataSet {
      std::vector<Data> set;
      std::vector<std::string> column_names;
      std::vector<Data>::iterator iterator;
      std::vector<Data>::iterator begin() { return set.begin(); }
      std::vector<Data>::iterator end() { return set.end(); }
      int size() { return set.size(); }
      void initializeIterator() { iterator = set.begin(); }
};

#endif
