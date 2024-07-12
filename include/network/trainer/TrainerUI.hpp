#ifndef __TRAINER_UI_HPP__
#define __TRAINER_UI_HPP__

#include "../../../libs/ProgressBar/ProgressBar.hpp"

class TrainerUI {
    public:
      TrainerUI(int total_epochs);
      void updateBar(int epochs_p, double loss, double validation_loss);
      void displayElapsedTime(double segs);
      void endBar();

    private:
      ProgressBar progress_bar;
      int total_epochs;
};

#endif
