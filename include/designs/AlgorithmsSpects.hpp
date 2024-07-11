#ifndef __ALGORITHMS_SPECTS_HPP__
#define __ALGORITHMS_SPECTS_HPP__

#include "../algorithms/AlphaAlgoritms.hpp"
struct AlgorithmsSpects {
      enum class AlphaModifier { WARMUP, DECAY, UNDEFINED };
      AlphaModifier alphaModifier = AlgorithmsSpects::AlphaModifier::UNDEFINED;
      AlphaAlgorithms::Arguments args_alpha_modifier;
};
#endif
