#ifndef __ALGORITHMS_SPECTS_HPP__
#define __ALGORITHMS_SPECTS_HPP__

#include "../../algorithms/AlphaAlgoritms.hpp"
#include "AlphaAlgoritmsSpects.hpp"
#include "EarlyStopSpects.hpp"
#include <memory>
struct AlgorithmsSpects {
      enum class AlphaModifier { WARMUP, DECAY, UNDEFINED };
      AlphaModifier alphaModifier = AlgorithmsSpects::AlphaModifier::UNDEFINED;
      std::unique_ptr<AlphaAlgorithmsSpects> args_alpha_modifier;

      EarlyStopSpects earlystop_spects;
};
#endif
