#include "../../../include/network/operator/checkpoints.hpp"
#include "../../../include/alerts/handler.hpp"
#include "../../../include/alerts/messages.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <sstream>
std::string generateName() {
      std::time_t now = time(nullptr);
      std::tm *localTime = std::localtime(&now);
      std::ostringstream filenameStream;
      filenameStream << "checkpoint_" << (localTime->tm_year + 1900) << '-'
                     << (localTime->tm_mon + 1) << '-' << localTime->tm_mday
                     << '_' << localTime->tm_hour << '_' << localTime->tm_min
                     << '_' << localTime->tm_sec << ".ckpt";
      return filenameStream.str();
}
void Checkpoint::createCheckpoint(std::vector<Parameters> &&network_params,
                                  std::string dir, TrainSpects &train_spects,
                                  AlgorithmsSpects &algorithms_spects,
                                  TYPE_CKPT type_ckpt) {
      // guardar las especificaciones (cada algoritmo tendra su funcion
      // read_from_ckpt, for_save), tambien trainSpects
      dest = dir;
      std::string name;
      if (type_ckpt == TYPE_CKPT::SAVE)
            name = dest + "/" + generateName();
      else
            name = dest + "/temp_best.ckpt";
      std::ofstream checkpoint_file(name);
      if (checkpoint_file) {
            dumpParameters(checkpoint_file, std::move(network_params));
      } else {
            Handler::warning({"\ncheckpoint could not be saved"});
      }
      checkpoint_file.close();
      if (type_ckpt == TYPE_CKPT::SAVE)
            Messages::Message({"\ncheckpoint ", name, " created"});
}

std::vector<Parameters> Checkpoint::loadCheckpoint(std::string path) {
      auto network_params = readCheckpoint(path);
      return network_params;
}

void Checkpoint::dumpParameters(std::ofstream &checkpoint_file,
                                std::vector<Parameters> &&params) {
      for (auto &neuron_params : params) {
            for (auto &weight : neuron_params) {
                  checkpoint_file << weight << ',';
            }
            checkpoint_file << neuron_params.bias << '\n';
      }
}

std::vector<Parameters> Checkpoint::readCheckpoint(std::string path) {
      std::vector<Parameters> checkpoint_parameters;
      std::ifstream checkpoint_file(path);
      if (checkpoint_file) {
            std::string line;
            while (std::getline(checkpoint_file, line))
                  checkpoint_parameters.push_back(readParameters(line));
      } else {
            Handler::terminalUserError({"checkpoint not found"});
      }
      return checkpoint_parameters;
}

Parameters Checkpoint::readParameters(std::string line) {
      std::vector<double> weights;
      double bias;
      std::istringstream stream_token(line);
      std::string token;
      while (std::getline(stream_token, token, ',')) {
            if (std::all_of(token.begin(), token.end(), [](const int c) {
                      return std::isdigit(c) || c == '.' || c == '-' ||
                             c == 'e' || c == '+';
                })) {
                  weights.push_back(std::stod(token));
            } else {
                  Handler::terminalSystemError({"checkpoint file corrupted"});
            }
      }
      bias = weights.back();
      weights.pop_back();
      return Parameters{weights, bias};
}
