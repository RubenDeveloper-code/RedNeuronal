#ifndef __DATA_HPP__
#define __DATA_HPP__

#include <utility>
#include <vector>
struct Data {
      Data(){};
      Data(int szInput, int szOuput) : input(szInput), output(szOuput){};
      Data(std::vector<double> in, std::vector<double> out)
          : input{in}, output{out} {};
      std::vector<double> input;
      std::vector<double> output;
};

using OutputNetworkData = std::vector<double>;
using InputNetworkData = std::vector<double>;
struct PairOutputs {
      OutputNetworkData computed;
      OutputNetworkData desired;
};

#endif
