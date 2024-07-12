#ifndef __NETWORK_ALGORITHS_HPP__
#define __NETWORK_ALGORITHS_HPP__

#include "../network/resources/SharedResources.hpp"
#include <iostream>
namespace AlphaAlgorithms {
enum class TYPE { WARM_UP, DECAY_LEARNING_RATE };
struct Arguments {
      SharedResources *shared_resources;
      Arguments(double initial_alpha, double final_alpha, int limit_epochs,
                bool apply_on_reloadckpt = false, double mod_recall = 0)
          : initial_alpha(initial_alpha), final_alpha(final_alpha),
            limit_epochs(limit_epochs),
            apply_on_reloadckpt(apply_on_reloadckpt), mod_recall(mod_recall) {}
      double initial_alpha;
      double final_alpha;
      int limit_epochs;
      bool apply_on_reloadckpt;
      double mod_recall;
      bool recalled = false;
};
struct AlphaAlgorithm {
      AlphaAlgorithm(AlphaAlgorithms::Arguments args) : args(args){};

      virtual void run() = 0;
      virtual ~AlphaAlgorithm() = default;
      void modifyLimits(int dir) {
            args.initial_alpha += (args.mod_recall * dir);
            args.final_alpha += (args.mod_recall * dir);
      }
      void recall() { args.recalled = true; }
      Arguments args;
};
struct WarmUp : public AlphaAlgorithm {
      WarmUp(AlphaAlgorithms::Arguments args) : AlphaAlgorithm(args){};
      void run() {
            if (args.recalled && args.apply_on_reloadckpt) {
                  modifyLimits(+1);
                  args.recalled = false;
            }
            if (*args.shared_resources->epochs_it <= args.limit_epochs)
                  *args.shared_resources->alpha =
                      args.initial_alpha +
                      (*args.shared_resources->epochs_it /
                       static_cast<float>(args.limit_epochs)) *
                          (args.final_alpha - args.initial_alpha);
            else {
                  *args.shared_resources->alpha = args.final_alpha;
            }
      }
};
struct DecayLearningRate : public AlphaAlgorithm {
      DecayLearningRate(AlphaAlgorithms::Arguments args)
          : AlphaAlgorithm(args){};
      void run() {
            if (args.recalled && args.apply_on_reloadckpt) {
                  modifyLimits(-1);
                  args.recalled = false;
            }
            if (*args.shared_resources->epochs_it <= args.limit_epochs)
                  *args.shared_resources->alpha =
                      args.initial_alpha +
                      ((*args.shared_resources->epochs_it /
                        static_cast<float>(args.limit_epochs)) *
                       (args.final_alpha - args.initial_alpha));
            else {
                  *args.shared_resources->alpha = args.final_alpha;
            }
            std::cout << "        alpha" << *args.shared_resources->alpha;
      }
};
std::unique_ptr<AlphaAlgorithm> newInstance(TYPE type, Arguments args);
} // namespace AlphaAlgorithms

#endif
