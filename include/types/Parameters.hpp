#ifndef __PARAMETER_HPP__
#define __PARAMETER_HPP__

#include <vector>
class Parameters {
    public:
      Parameters(std::vector<double> w, double b) : weights(w), bias(b) {}
      std::vector<double> weights;
      double bias;
      std::vector<double>::iterator begin() { return weights.begin(); }
      std::vector<double>::iterator end() { return weights.end(); }
};

#endif
