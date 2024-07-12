#ifndef __ALGORITHMS_SPECTS_HPP__
#define __ALGORITHMS_SPECTS_HPP__

#include "../../algorithms/AlphaAlgoritms.hpp"
#include <memory>
struct AlgorithmsSpects {
      enum class AlphaModifier { WARMUP, DECAY, UNDEFINED };
      AlphaModifier alphaModifier = AlgorithmsSpects::AlphaModifier::UNDEFINED;
      std::unique_ptr<AlphaAlgorithms::Arguments> args_alpha_modifier;
};
#endif
