#include "../../../include/network/trainer/Checks.hpp"
#include "../../../include/alerts/messages.hpp"
#include <iostream>

Checks::Checks(Network &network) : network(network) {}

bool Checks::reachUmbrall(double loss, double umbral, double epoch) {
      if (loss < umbral) {
            Messages::Message({"done in epoch: ", std::to_string(epoch),
                               " loss: ", std::to_string(loss)});
            return true;
      }
      return false;
}
