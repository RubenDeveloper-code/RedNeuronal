#ifndef __CHECKPOINTS_HPP__
#define __CHECKPOINTS_HPP__

#include "../../include/core/Network.hpp"
#include <fstream>
#include <string>
#include <vector>
struct Checkpoint {
      enum class TYPE_CKPT { TEMP, SAVE, SKIP } type_ckpt;
      Checkpoint(){};
      Checkpoint(std::string dest);
      void createCheckpoint(std::vector<Parameters> &&network_params,
                            std::string dir, TYPE_CKPT type_ckpt);
      std::vector<Parameters> loadCheckpoint(std::string path);

    private:
      std::string dest;
      void dumpParameters(std::ofstream &checkpoint_file,
                          std::vector<Parameters> &&params);
      std::vector<Parameters> readCheckpoint(std::string path);
      Parameters readParameters(std::string line);
};
#endif
