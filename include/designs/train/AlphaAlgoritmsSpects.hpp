#ifndef __AAA_SPECTS__
#define __AAA_SPECTS__

#include "../../network/resources/SharedResources.hpp"
class AlphaAlgorithmsSpects {
    public:
      SharedResources *shared_resources;
      AlphaAlgorithmsSpects(double initial_alpha, double final_alpha,
                            int limit_epochs, bool apply_on_reloadckpt = false,
                            double mod_recall = 0)
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

#endif
