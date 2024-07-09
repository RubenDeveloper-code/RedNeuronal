#ifndef __NETWORK_CONSTRUCTOR_HPP__
#define __NETWORK_CONSTRUCTOR_HPP__

#include "../designs/ModelDesign.hpp"
#include "Network.hpp"
#include "SharedResources.hpp"
#include <memory>
class NetworkConstructor {
    public:
      void construct(Network &network, ModelDesign &modelDesign,
                     SharedResources &shared_res,
                     std::shared_ptr<LossFuctions::LossFunction> loss_function);

    private:
      void
      createLayers(Network &network, ModelDesign &model_design,
                   SharedResources &shared_res,
                   std::shared_ptr<LossFuctions::LossFunction> loss_function);
      void connectLayers(Network &network);
};

#endif
