#ifndef __CHECKPOINT_SECTS_HPP__
#define __CHECKPOINT_SECTS_HPP__
#include "string"
struct CheckpointSpects {
      const int checkpoint_frec;
      const std::string checkpoints_folder;
      const std::string tempcheckpoints_folder = "../temp";
};
#endif
