#pragma once
#include <iostream>
#include <memory>
#include <stdio.h>
#include <string>
#include <vector>

class ProgressBar {
    public:
      /* Takes in a char for filling up the bar and the size fo the bar */
      ProgressBar(char notDoneChar, char doneChar, unsigned size);
      void end();

      unsigned int todo;
      unsigned int done;

      /* Fills the bar upto a given number */
      void fillUpCells(unsigned int cells);

      /* Fills the bar one by one */
      void fillUp(int actual_val);

      /* Displays the percentage beside the bar */
      void displayPercentage();

      bool time_to_fill(int actual_val);
      /* Shows tasks done out of the tasks to be done */
      void displayTasksDone();

      void displayProgress(double progress, double total);

      void endlBar();

      void displayTrail(std::string trail_name, float trail);

      void displayElapsedTime(int h, int m, int s);
      /* Returns the size of the progress bar */
      unsigned int getSize();

    private:
      unsigned int size = 0;
      unsigned int pos = 1;
      unsigned int max;
      char c;
      char ch;
      bool show = false;
      std::vector<char> bar;
};
