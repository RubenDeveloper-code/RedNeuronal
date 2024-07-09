#ifndef __NETWORK_ALGORITHS_HPP__
#define __NETWORK_ALGORITHS_HPP__

#include "../core/SharedResources.hpp"
#include <iostream>
namespace AlphaAlgorithms {
enum class TYPE { WARM_UP, DECAY_LEARNING_RATE };
struct Arguments {
      SharedResources *shared_resources;
      double initial_alpha;
      double final_alpha;
      int limit_epochs;
};
struct AlphaAlgorithm {
      AlphaAlgorithm(AlphaAlgorithms::Arguments args) : args(args){};

      virtual void run() = 0;
      virtual ~AlphaAlgorithm() = default;
      Arguments args;
};
struct WarmUp : public AlphaAlgorithm {
      WarmUp(AlphaAlgorithms::Arguments args) : AlphaAlgorithm(args){};
      void run() {
            if (*args.shared_resources->epochs_it <= args.limit_epochs)
                  *args.shared_resources->alpha =
                      args.initial_alpha +
                      (*args.shared_resources->epochs_it /
                       static_cast<float>(args.limit_epochs)) *
                          (args.final_alpha - args.initial_alpha);
            else
                  *args.shared_resources->alpha = args.final_alpha;
      }
};
struct DecayLearningRate : public AlphaAlgorithm {
      DecayLearningRate(AlphaAlgorithms::Arguments args)
          : AlphaAlgorithm(args){};
      void run() {
            if (*args.shared_resources->epochs_it <= args.limit_epochs)
                  *args.shared_resources->alpha =
                      args.initial_alpha -
                      (*args.shared_resources->epochs_it /
                       static_cast<float>(args.limit_epochs)) *
                          (args.final_alpha - args.initial_alpha);
            else
                  *args.shared_resources->alpha = args.final_alpha;
      }
};
std::unique_ptr<AlphaAlgorithm> newInstance(TYPE type, Arguments args);
} // namespace AlphaAlgorithms

#endif
