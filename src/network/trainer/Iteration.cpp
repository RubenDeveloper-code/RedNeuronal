#include "../../../include/network/trainer/Iteration.hpp"

Iteration::Iteration(Network &network, SetterData &&setter_data,
                     ModelLoss &model_loss, unsigned minibatch)
    : network(network), setter_data(std::move(setter_data)),
      model_loss(model_loss), minibatch_size(minibatch){};

double Iteration::iterate() {
      std::vector<PairOutputs> pair_outputs;
      const int limit = setter_data.getDataSize() + minibatch_size;
      int minibatch_proccessed{};
      double sum_loss = 0;
      for (int multi = 1; multi * minibatch_size < limit; multi++) {
            pair_outputs = forwardPropagation();
            backpropagation(pair_outputs);
            auto loss = model_loss.computeLoss(pair_outputs);
            sum_loss += loss;
            minibatch_proccessed++;
      }
      double loss = sum_loss / minibatch_proccessed;
      return loss;
}

std::vector<PairOutputs> Iteration::forwardPropagation() {
      std::vector<PairOutputs> pair_outputs;
      for (auto i = 0; i < minibatch_size; i++) {
            auto pair_output = individualDataForewardPropagation();
            pair_outputs.push_back(pair_output);
      }
      return pair_outputs;
}

PairOutputs Iteration::individualDataForewardPropagation() {
      auto desired =
          setter_data.prepareNextEpoch(network.input(), network.ouput());
      auto output = network_operator.computeNetworkOutput(network);
      return PairOutputs{output, desired.output};
}

void Iteration::backpropagation(std::vector<PairOutputs> mini_batch_outputs) {
      network_operator.recalculateNetworkParameters(network,
                                                    mini_batch_outputs);
}

double Iteration::validationForwardPropagation(DataSet &validation_data) {
      SetterData setter(validation_data);
      std::vector<PairOutputs> outputValidation;
      for (auto &data : validation_data) {
            setter.preparePrediction(network.input(), data.input);
            auto output = network_operator.computeNetworkOutput(network);
            outputValidation.push_back(PairOutputs{output, data.output});
      }
      return model_loss.computeLoss(outputValidation);
}
