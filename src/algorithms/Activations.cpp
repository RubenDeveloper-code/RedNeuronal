#include "../../include/algorithms/Activations.hpp"

#include <memory>

std::shared_ptr<Activations::activation>
Activations::newInstance(Activations::TYPE type) {
      switch (type) {
      case Activations::TYPE::SIGMOID:
            return std::make_shared<Activations::sigmoid>();
      case Activations::TYPE::RELU:
            return std::make_shared<Activations::relu>();
      case Activations::TYPE::REGRESSION:
            return std::make_shared<Activations::regression>();
      }
      return std::make_shared<Activations::regression>();
}
