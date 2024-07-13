#ifndef __PATIENCE_HPP__
#define __PATIENCE_HPP__

struct EarlyStopSpects {
      EarlyStopSpects() = default;
      EarlyStopSpects(unsigned patience, bool earlyStop_restart)
          : patience(patience), earlyStop_restart(earlyStop_restart),
            active(true){};
      unsigned patience;
      bool earlyStop_restart;
      bool active = false;
};

#endif
