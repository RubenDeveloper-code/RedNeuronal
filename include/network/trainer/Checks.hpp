#ifndef __PERFOMANCE_HPP__
#define __PERFOMANCE_HPP__

#include "../../designs/train/EarlyStopSpects.hpp"
#include "../Network.hpp"

class Checks {
    public:
      Checks(Network &network);
      bool reachUmbrall(double loss, double umbral, double epoch);

    private:
      Network &network;
};
#endif
