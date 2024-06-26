#ifndef __NETGLOBALRESOPURSES__
#define __NETGLOBALRESOPURSES__
#include <memory>
struct GlobalResourses {
      GlobalResourses(double initialAlpha) {
            epochs_it = std::make_shared<int>(0);
            alpha = std::make_shared<double>(initialAlpha);
      }
      std::shared_ptr<int> epochs_it;
      std::shared_ptr<double> alpha;
};
#endif
