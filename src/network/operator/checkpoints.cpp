#include "../../../include/network/operator/checkpoints.hpp"
#include "../../../include/alerts/handler.hpp"
#include "../../../include/alerts/messages.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <memory>
#include <ranges>
#include <sstream>
#include <string>
#include <vector>

#define INPUT 1
#define OUTPUT 2
#define EPOCH 0

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
                                  int sizeInput, int sizeOutput,
                                  int actual_epoch, TYPE_CKPT type_ckpt) {
      dest = dir;
      std::string name;
      if (type_ckpt == TYPE_CKPT::SAVE)
            name = dest + "/" + generateName();
      else
            name = dest + "/temp_best.ckpt";
      std::ofstream checkpoint_file(name);
      buildCkptHeader(checkpoint_file, actual_epoch, sizeInput, sizeOutput);
      if (checkpoint_file) {
            dumpParameters(checkpoint_file, std::move(network_params));
      } else {
            Handler::warning({"\ncheckpoint could not be saved"});
      }
      checkpoint_file.close();
      if (type_ckpt == TYPE_CKPT::SAVE)
            Messages::Message({"\ncheckpoint ", name, " created"});
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

std::vector<Parameters>
Checkpoint::loadCheckpoint(std::string path, int sizeInput, int sizeOutput,
                           std::shared_ptr<int> epoch_it) {
      std::ifstream checkpoint_file(path);
      *epoch_it = loadCkptHeader(checkpoint_file, sizeInput, sizeOutput);
      auto network_params = readCheckpoint(checkpoint_file);
      checkpoint_file.close();
      return network_params;
}

std::vector<Parameters>
Checkpoint::readCheckpoint(std::ifstream &checkpoint_file) {
      std::vector<Parameters> checkpoint_parameters;
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
void Checkpoint::buildCkptHeader(std::ofstream &out, int epoch_it,
                                 int sizeInput, int sizeOutput) {
      out << epoch_it << "," << sizeInput << "," << sizeOutput << '\n';
}
int Checkpoint::loadCkptHeader(std::ifstream &in, int sizeInput,
                               int sizeOutput) {
      std::vector<int> header_tokens;
      std::string header;
      std::getline(in, header);
      std::istringstream stream_header(header);
      std::string token;
      while (std::getline(stream_header, token, ',') && token != "\n") {
            if (std::all_of(token.begin(), token.end(), [](const int c) {
                      return std::isdigit(c) || c == '.' || c == '-' ||
                             c == 'e' || c == '+';
                })) {
                  header_tokens.push_back(std::stod(token));
            } else {
                  Handler::terminalSystemError(
                      {"checkpoint file corrupted (header)"});
            }
      }
      if (header_tokens.size() != 3)
            Handler::terminalSystemError(
                {"checkpoint file corrupted (header args)"});

      if (header_tokens[INPUT] == sizeInput &&
          header_tokens[OUTPUT] == sizeOutput)
            return header_tokens[EPOCH];
      Handler::terminalUserError({"Inputs and Outputs do not match"});
      return 0;
}
