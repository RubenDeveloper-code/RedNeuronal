#include "ProgressBar.hpp"
#include <memory>
/* Defining the constructor */
ProgressBar::ProgressBar(char notDoneChar, char doneChar, unsigned int _size)
    : c(doneChar), ch(notDoneChar), size(100), todo(0), done(0) {
      bar.push_back('[');
      max = _size;
      for (int i = 1; i < size + 1; i++) {
            bar.push_back(ch);
      }

      bar.push_back(']');
}
/* Defining fillUpCells */
void ProgressBar::fillUpCells(unsigned int cells) {
      pos = 0;
      for (int i = 1; i < cells; i++) {
            bar[i] = c;
            std::cout << '\r';
            for (int j = 0; j < bar.size(); j++) {
                  std::cout << bar[j] << std::flush;
            }
      }
      pos += cells;
}
bool ProgressBar::time_to_fill(int actual_val) {
      if (actual_val == -1)
            return true;
      float a = ((float)actual_val / max) * 100;
      return pos < a;
}
/* Defining fillUp */
void ProgressBar::fillUp(int actual_val) {
      show = false;
      if (time_to_fill(actual_val)) {
            show = true;
            bar[pos] = c;
            pos++;
      }
      std::cout << '\r';

      for (int i = 0; i < bar.size(); i++) {
            std::cout << bar[i] << std::flush;
      }
}

/* Displays the percentage beside the bar */
void ProgressBar::displayPercentage() {
      float percent = ((float)pos / (float)(bar.size() - 1)) * 100;
      std::cout << (int)percent << "%";
}

void ProgressBar::displayProgress(double progress) {
      std::cout << " epoch: " << progress << " ";
}
/* Shows tasks done out of the tasks to be done */
void ProgressBar::displayTasksDone() {
      std::cout << '(' << done << '/' << todo << ')' << std::flush;
}

void ProgressBar::displayTrail(float trail) {
      std::cout << "| loss: " << trail << std::flush;
}

/* Returns the size of the progress bar */

unsigned int ProgressBar::getSize() { return bar.size() - 2; }

void ProgressBar::end() { std::cout << std::endl; }
