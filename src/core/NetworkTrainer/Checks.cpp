#include "../../../include/core/NetworkTrainer/Checks.hpp"
#include "../../../include/alerts/messages.hpp"
#include <iostream>

Checks::Checks(Network &network, EarlyStopSpects &earlystop_spects)
    : network(network), earlystop_spects(earlystop_spects) {}

bool Checks::reachUmbrall(double loss, double umbral, double epoch) {
      if (loss < umbral) {
            Messages::Message({"done in epoch: ", std::to_string(epoch),
                               " loss: ", std::to_string(loss)});
            return true;
      }
      return false;
}
// se llama desde fit
bool Checks::reachPatience(double validation_loss) {
      static unsigned actual_patience;
      static double best_validation_loss = validation_loss;
      if (validation_loss < best_validation_loss) {
            actual_patience = 0;
            best_validation_loss = validation_loss;
      } else
            actual_patience++;
      if (actual_patience == earlystop_spects.patience) {
            actual_patience = 0;
            return true;
      }
      return false;
}
