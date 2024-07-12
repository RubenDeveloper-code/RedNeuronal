#include "../../../include/network/trainer/TrainerUI.hpp"

TrainerUI::TrainerUI(int total_epochs)
    : progress_bar(' ', '#', total_epochs), total_epochs(total_epochs) {}

void TrainerUI::updateBar(int epochs_p, double loss, double validation_loss) {
      static double p_loss, p_validation;
      std::string c;
      progress_bar.fillUp(epochs_p);
      progress_bar.displayProgress(epochs_p, total_epochs);
      c = (loss < p_loss) ? "\033[32m-\033[0m" : "\033[31m+\033[0m";
      progress_bar.displayTrail(c + "loss", loss);
      c = (validation_loss < p_validation) ? "\033[32m-\033[0m"
                                           : "\033[31m+\033[0m";
      progress_bar.displayTrail(c + "validation loss", validation_loss);
      p_validation = validation_loss;
      p_loss = loss;
}

void TrainerUI::displayElapsedTime(double segs) {
      int horas = (int)segs / 3600;
      int mins = ((int)segs % 3600) / 60;
      int seconds = (int)segs % 60;
      progress_bar.displayElapsedTime(horas, mins, seconds);
      progress_bar.endlBar();
}

void TrainerUI::endBar() { progress_bar.end(); }
