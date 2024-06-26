#ifndef __DATA_HPP__
#define __DATA_HPP__

#include <vector>
struct Data {
      Data(int szInput, int szOuput) : input(szInput), output(szOuput){};
      Data(std::vector<double> in, std::vector<double> out)
          : input{in}, output{out} {};
      std::vector<double> input;
      std::vector<double> output;
};
using NetworkTrainData = std::vector<Data>;
using InputNetworkData = std::vector<double>;
using OutputNetworkData = std::vector<double>;
#endif
