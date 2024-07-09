#ifndef __NETGLOBALRESOPURSES__
#define __NETGLOBALRESOPURSES__
#include <memory>
struct SharedResources {
      SharedResources() {
            epochs_it = std::make_shared<int>(0);
            alpha = std::make_shared<double>();
      }
      void init(double _alpha) { *alpha = _alpha; }
      std::shared_ptr<int> epochs_it;
      std::shared_ptr<double> alpha;
};
#endif
