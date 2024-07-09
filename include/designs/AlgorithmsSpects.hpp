#ifndef __ALGORITHMS_SPECTS_HPP__
#define __ALGORITHMS_SPECTS_HPP__

#include "../algorithms/AlphaAlgoritms.hpp"
struct AlgorithmsSpects {
      enum class AlphaModifier { WARMUP, DECAY };
      AlphaModifier alphaModifier;
      AlphaAlgorithms::Arguments args_alpha_modifier;
};
#endif
