#include "../../include/algorithms/AlphaAlgoritms.hpp"
#include <memory>

std::unique_ptr<AlphaAlgorithms::AlphaAlgorithm>
AlphaAlgorithms::newInstance(AlphaAlgorithms::TYPE type,
                             AlphaAlgorithmsSpects args) {
      switch (type) {
      case AlphaAlgorithms::TYPE::WARM_UP:
            return std::make_unique<AlphaAlgorithms::WarmUp>(args);
      case AlphaAlgorithms::TYPE::DECAY_LEARNING_RATE:
            return std::make_unique<AlphaAlgorithms::DecayLearningRate>(args);
      }
      return nullptr;
}
