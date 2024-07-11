#include "../../include/core/NetworkTrainer.hpp"
#include "../../include/alerts/messages.hpp"
#include "../../include/data/checkpoints.hpp"
#include "../../libs/ProgressBar.hpp"
#include <algorithm>
#include <chrono>
#include <memory>
#include <vector>

NetworkTrainer::Status
NetworkTrainer::fit(Network &network, SharedResources &shared_resources,
                    TrainSpects train_spects,
                    AlgorithmsSpects &algorithms_spects,
                    NetworkOperator &network_operator,
                    std::shared_ptr<LossFuctions::LossFunction> loss_function) {
      setter_data =
          std::move(SetterData(train_spects.dataset.training_dataset));
      double loss, validation_loss;
      std::string dest_folder_ckpt;
      initAlgorithms(algorithms_spects);
      status = Status::FITTING;
      Messages::Message({"training model..."});
      ProgressBar bar(' ', '#', train_spects.epochs);
      while ((*shared_resources.epochs_it)++ < train_spects.epochs) {
            auto start = std::chrono::high_resolution_clock::now();
            if (alpha_algorithm != nullptr)
                  alpha_algorithm->run();
            loss = iteration(network, train_spects, network_operator,
                             loss_function);
            validation_loss = checkValidationLoss(
                network, network_operator,
                train_spects.dataset.validation_dataset, loss_function);
            updateBar(bar, *shared_resources.epochs_it, train_spects.epochs,
                      loss, validation_loss);
            adminCheckpoint(network, network_operator, train_spects,
                            *shared_resources.epochs_it, validation_loss);
            if (status != Status::FITTING) {
                  bar.end();
                  return status;
            }
            auto end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(end - start).count();
            displayElapsedTime(bar, elapsed * train_spects.epochs);
      }
      bar.end();
      Messages::Message({"trained model"});
      return Status::DONE;
}
double NetworkTrainer::iteration(
    Network &network, TrainSpects &train_spects,
    NetworkOperator &network_operator,
    std::shared_ptr<LossFuctions::LossFunction> loss_function) {
      std::vector<PairOutputs> pair_outputs;
      const int limit = setter_data.getDataSize() + train_spects.mini_batch;
      double sum_loss = 0;
      for (int multi = 1; multi * train_spects.mini_batch < limit; multi++) {
            pair_outputs = forwardPropagation(network, network_operator,
                                              train_spects.mini_batch);
            backpropagation(network, network_operator, pair_outputs);
            auto loss = computeLoss(pair_outputs, loss_function);
            sum_loss += loss;
      }
      double loss = sum_loss / train_spects.mini_batch;
      if (checkIfReachUmbrall(loss, train_spects.umbral, train_spects.epochs))
            status = Status::DONE;
      return loss;
}

std::vector<PairOutputs> NetworkTrainer::forwardPropagation(
    Network &network, NetworkOperator &network_operator, double mini_batch) {
      std::vector<PairOutputs> pair_outputs;
      for (auto i = 0; i < mini_batch; i++) {
            auto pair_output =
                individualDataForewardPropagation(network, network_operator);
            pair_outputs.push_back(pair_output);
      }
      return pair_outputs;
}

PairOutputs NetworkTrainer::individualDataForewardPropagation(
    Network &network, NetworkOperator &network_operator) {
      auto desired =
          setter_data.prepareNextEpoch(network.input(), network.ouput());
      auto output = network_operator.computeNetworkOutput(network);
      return PairOutputs{output, desired.output};
}

void NetworkTrainer::backpropagation(
    Network &network, NetworkOperator &network_operator,
    std::vector<PairOutputs> mini_batch_outputs) {
      network_operator.recalculateNetworkParameters(network,
                                                    mini_batch_outputs);
}

double NetworkTrainer::checkValidationLoss(
    Network &network, NetworkOperator &network_operator,
    DataSet &validation_data,
    std::shared_ptr<LossFuctions::LossFunction> loss_function) {
      auto validation_batch_loss =
          validationOutputs(network, network_operator, validation_data);
      double validation_loss =
          computeLoss(validation_batch_loss, loss_function);
      return validation_loss;
}
std::vector<PairOutputs>
NetworkTrainer::validationOutputs(Network &network,
                                  NetworkOperator &network_operator,
                                  DataSet &validation_data) {
      SetterData setter(validation_data);
      std::vector<PairOutputs> outputValidation;
      for (auto &data : validation_data) {
            setter.preparePrediction(network.input(), data.input);
            auto output = network_operator.computeNetworkOutput(network);
            outputValidation.push_back(PairOutputs{output, data.output});
      }
      return outputValidation;
}

bool NetworkTrainer::checkIfReachUmbrall(double loss, double umbral,
                                         double epoch) {
      if (loss < umbral) {
            Messages::Message({"done in epoch: ", std::to_string(epoch),
                               " loss: ", std::to_string(loss)});
            return true;
      }
      return false;
}

void NetworkTrainer::initAlgorithms(AlgorithmsSpects &algorithms_spects) {
      alpha_algorithm = AlphaAlgorithms::newInstance(
          static_cast<AlphaAlgorithms::TYPE>(algorithms_spects.alphaModifier),
          algorithms_spects.args_alpha_modifier);
}

double NetworkTrainer::computeLoss(
    std::vector<PairOutputs> pair_outputs,
    std::shared_ptr<LossFuctions::LossFunction> loss_function) {
      auto perfomance = minibatchPerfomance(pair_outputs);
      double loss =
          loss_function->function(perfomance.computed, perfomance.desired);
      return loss;
}
PairOutputs NetworkTrainer::minibatchPerfomance(
    std::vector<PairOutputs> minibatchs_perfomance) {
      PairOutputs perfomance;
      for (auto mini_batch : minibatchs_perfomance) {
            std::for_each(mini_batch.computed.begin(),
                          mini_batch.computed.end(),
                          [&perfomance](double computed) {
                                perfomance.computed.push_back(computed);
                          });
            std::for_each(mini_batch.desired.begin(), mini_batch.desired.end(),
                          [&perfomance](double desired) {
                                perfomance.desired.push_back(desired);
                          });
      }
      return perfomance;
}
// crear constro central de entrenamiento alv
void NetworkTrainer::adminCheckpoint(Network &network,
                                     NetworkOperator &network_operator,
                                     TrainSpects &train_spects, int a_epoch,
                                     double validation_loss) {
      static unsigned actual_patience;
      static double best_validation_loss = validation_loss;
      std::string dest_folder;
      Checkpoint::TYPE_CKPT type_ckpt;
      if (validation_loss < best_validation_loss) {
            actual_patience = 0;
            best_validation_loss = validation_loss;
            if (checkIfCheckpoint(a_epoch, train_spects.checkpoint_frec)) {
                  dest_folder = train_spects.checkpoints_folder;
                  type_ckpt = Checkpoint::TYPE_CKPT::SAVE;
            } else {
                  dest_folder = train_spects.tempcheckpoints_folder;
                  type_ckpt = Checkpoint::TYPE_CKPT::TEMP;
            }
            saveCheckpoint(network, network_operator, dest_folder, type_ckpt);
      } else
            actual_patience++;
      if (actual_patience == train_spects.patience)
            earlyStop(train_spects.earlyStop_restart);
}
void NetworkTrainer::saveCheckpoint(Network &network,
                                    NetworkOperator &network_operator,
                                    std::string dest_folder,
                                    Checkpoint::TYPE_CKPT type_ckpt) {
      Checkpoint ckpt;
      auto network_params = network_operator.getNetworkParameters(network);
      ckpt.createCheckpoint(std::move(network_params), dest_folder, type_ckpt);
}

void NetworkTrainer::earlyStop(bool restart) {
      if (restart) {
            Messages::Message({"\nRollback last best trained model"});
            status = Status::RELOAD;
      } else {
            Messages::Message({"\nPatience reached"});
            status = Status::DONE;
      }
}

void NetworkTrainer::updateBar(ProgressBar &progress_bar, int epochs_p,
                               int epochs, double loss,
                               double validation_loss) {
      static double p_loss, p_validation;
      std::string c;
      progress_bar.fillUp(epochs_p);
      progress_bar.displayProgress(epochs_p, epochs);
      c = (loss < p_loss) ? "\033[32m-\033[0m" : "\033[31m+\033[0m";
      progress_bar.displayTrail(c + "loss", loss);
      c = (validation_loss < p_validation) ? "\033[32m-\033[0m"
                                           : "\033[31m+\033[0m";
      progress_bar.displayTrail(c + "validation loss", validation_loss);
      p_validation = validation_loss;
      p_loss = loss;
}

void NetworkTrainer::displayElapsedTime(ProgressBar &bar, double segs) {
      int horas = (int)segs / 3600;
      int mins = ((int)segs % 3600) / 60;
      int seconds = (int)segs % 60;
      bar.displayElapsedTime(horas, mins, seconds);
      bar.endlBar();
}
bool NetworkTrainer::checkIfCheckpoint(int a_epoch, int frec) {
      return a_epoch % frec == 0;
}
