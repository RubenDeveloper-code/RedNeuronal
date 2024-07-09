#include "../../include/core/NetworkTrainer.hpp"
#include "../../include/alerts/messages.hpp"
#include "../../libs/ProgressBar.hpp"
#include <algorithm>
#include <memory>
#include <vector>

enum class Status { FITTING, DONE } status;

void NetworkTrainer::fit(
    Network &network, SharedResources &shared_resources,
    TrainSpects train_spects, AlgorithmsSpects &algorithms_spects,
    NetworkOperator &network_operator,
    std::shared_ptr<LossFuctions::LossFunction> loss_function) {
      setter_data = std::move(SetterData(train_spects.dataset));
      double lastloss;
      initAlgorithms(algorithms_spects);
      status = Status::FITTING;
      Messages::Message({"training model"});
      ProgressBar bar(' ', '#', train_spects.epochs);
      while ((*shared_resources.epochs_it)++ < train_spects.epochs) {
            if (alpha_algorithm != nullptr)
                  alpha_algorithm->run();
            lastloss = iteration(network, train_spects, network_operator,
                                 loss_function);
            if (status == Status::DONE)
                  return;
            bar.fillUp(*shared_resources.epochs_it);
            bar.displayProgress(*shared_resources.epochs_it);
            bar.displayTrail(lastloss);
      }
      bar.end();
      Messages::Message({"trained model"});
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
