#include "../../../include/network/trainer/TrainerAlgorithms.hpp"

TrainerAlgorithms::TrainerAlgorithms(AlgorithmsSpects &algorithms_spects,
                                     SharedResources &shared_resources)
    : algorithms_spects(algorithms_spects), shared_resources(shared_resources) {
      initAlgorithms(algorithms_spects);
}

void TrainerAlgorithms::runAlphaAlgorithm() {
      if (algorithms_spects.alphaModifier !=
          AlgorithmsSpects::AlphaModifier::UNDEFINED)
            alpha_algorithm->run();
}
bool TrainerAlgorithms::runEarlyStopAdmin(double validation_loss) {
      if (algorithms_spects.earlystop_spects.active) {
            static unsigned actual_patience;
            static double best_validation_loss = validation_loss;
            if (validation_loss < best_validation_loss) {
                  actual_patience = 0;
                  best_validation_loss = validation_loss;
            } else
                  actual_patience++;
            if (actual_patience ==
                algorithms_spects.earlystop_spects.patience) {
                  actual_patience = 0;
                  return true;
            }
      }
      return false;
}

void TrainerAlgorithms::initAlgorithms(AlgorithmsSpects &algorithms_spects) {
      if (algorithms_spects.alphaModifier !=
          AlgorithmsSpects::AlphaModifier::UNDEFINED) {
            algorithms_spects.args_alpha_modifier->shared_resources =
                &shared_resources;
            alpha_algorithm = AlphaAlgorithms::newInstance(
                static_cast<AlphaAlgorithms::TYPE>(
                    algorithms_spects.alphaModifier),
                *algorithms_spects.args_alpha_modifier);
      }
}

void TrainerAlgorithms::restart() {
      if (algorithms_spects.alphaModifier !=
          AlgorithmsSpects::AlphaModifier::UNDEFINED)
            if (algorithms_spects.args_alpha_modifier->apply_on_reloadckpt)
                  alpha_algorithm->recall();
}
