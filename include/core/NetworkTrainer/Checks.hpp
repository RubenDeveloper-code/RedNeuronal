#ifndef __PERFOMANCE_HPP__
#define __PERFOMANCE_HPP__

#include "../../designs/Train/EarlyStopSpects.hpp"
#include "../Network.hpp"

class Checks {
    public:
      Checks(Network &network, EarlyStopSpects &earlystop_spects);
      bool reachUmbrall(double loss, double umbral, double epoch);
      bool reachPatience(double validation_loss);

    private:
      Network &network;
      EarlyStopSpects &earlystop_spects;
};
#endif
