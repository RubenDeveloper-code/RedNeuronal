#ifndef __CHECKPOINTS_HPP__
#define __CHECKPOINTS_HPP__

#include "../../designs/train/AlgorithmsSpects.hpp"
#include "../../designs/train/TrainSpects.hpp"
#include "../Network.hpp"
#include <fstream>
#include <string>
#include <vector>
struct Checkpoint {
      enum class TYPE_CKPT { TEMP, SAVE, SKIP } type_ckpt;
      Checkpoint(){};
      Checkpoint(std::string dest);
      void createCheckpoint(std::vector<Parameters> &&network_params,
                            std::string dir, TrainSpects &train_spects,
                            int sizeInput, int sizeOutput, int actual_epoch,
                            TYPE_CKPT type_ckpt);
      std::vector<Parameters> loadCheckpoint(std::string path, int sizeInput,
                                             int sizeOutput,
                                             std::shared_ptr<int> epoch_it);

    private:
      std::string dest;
      void dumpParameters(std::ofstream &checkpoint_file,
                          std::vector<Parameters> &&params);
      std::vector<Parameters> readCheckpoint(std::ifstream &checkpoint_file);
      Parameters readParameters(std::string line);
      void buildCkptHeader(std::ofstream &out, int epoch_it, int sizeInput,
                           int sizeOutput);
      int loadCkptHeader(std::ifstream &in, int sizeInput, int sizeOutput);
};
#endif
