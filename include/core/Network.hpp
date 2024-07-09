#ifndef __NETWORK_HPP__
#define __NETWORK_HPP__

#include "Layer.hpp"
#include <vector>
class Network {
    public:
      std::vector<Layer> layers;
      Layer &input() { return layers[0]; }
      Layer &ouput() { return layers[layers.size() - 1]; }
      std::vector<Layer>::iterator begin() { return layers.begin(); }
      std::vector<Layer>::iterator end() { return layers.end(); }
      std::vector<Layer>::reverse_iterator rbegin() { return layers.rbegin(); }
      std::vector<Layer>::reverse_iterator rend() { return layers.rend(); }

      int inputSize() { return input().neurons.size(); }
      int ouputSize() { return ouput().neurons.size(); }
};
#endif
