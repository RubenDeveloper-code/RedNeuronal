#ifndef __MODELDEDIGN_HPP__
#define __MODELDEDIGN_HPP__

#include "LayerDesign.hpp"
#include <vector>

class ModelDesign {
    public:
      std::vector<LayerDesign> design;
      enum class LossFuction {
            MEAN_SQUARED_ERROR,
            BINARY_CROSS_ENTROPY,
            UNDEFINED
      } loss_function = ModelDesign::LossFuction::UNDEFINED;
      bool checkIntegrity();

    private:
      bool sortLayerDesign();
};

#endif
