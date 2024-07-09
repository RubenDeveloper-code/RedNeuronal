#include "../../include/algorithms/Optimizers.hpp"
std::shared_ptr<Optimizers::Optimizer>
Optimizers::newInstance(TYPE type, SharedResources &shared_resources) {
      switch (type) {
      case TYPE::SGD:
            return std::make_shared<SDG>(shared_resources.alpha);
      case TYPE::ADAMS:
            return std::make_shared<Adams>(shared_resources.epochs_it,
                                           shared_resources.alpha);
      case TYPE::ADAGRAD:
            return std::make_shared<AdaGrad>(shared_resources.alpha);
      case TYPE::RMSPROP:
            return std::make_shared<RMSProp>(shared_resources.alpha);
      case TYPE::MOMENTUM:
            return std::make_shared<Momentum>(shared_resources.alpha);
      }
      return std::make_shared<SDG>(shared_resources.alpha);
}
