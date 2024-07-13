#ifndef __NETWORK_ALGORITHS_HPP__
#define __NETWORK_ALGORITHS_HPP__

#include "../designs/train/AlphaAlgoritmsSpects.hpp"
#include "../network/resources/SharedResources.hpp"
#include <iostream>
namespace AlphaAlgorithms {
enum class TYPE { WARM_UP, DECAY_LEARNING_RATE };
struct AlphaAlgorithm {
      AlphaAlgorithm(AlphaAlgorithmsSpects args) : args(args){};

      virtual void run() = 0;
      virtual ~AlphaAlgorithm() = default;
      void modifyLimits(int dir) {
            args.initial_alpha += (args.mod_recall * dir);
            args.final_alpha += (args.mod_recall * dir);
      }
      void recall() { args.recalled = true; }
      AlphaAlgorithmsSpects args;
};
struct WarmUp : public AlphaAlgorithm {
      WarmUp(AlphaAlgorithmsSpects args) : AlphaAlgorithm(args){};
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
      DecayLearningRate(AlphaAlgorithmsSpects args) : AlphaAlgorithm(args){};
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
std::unique_ptr<AlphaAlgorithm> newInstance(TYPE type,
                                            AlphaAlgorithmsSpects args);
} // namespace AlphaAlgorithms

#endif
