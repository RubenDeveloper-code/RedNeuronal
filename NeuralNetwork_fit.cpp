#include "NeuralNetwork_fit.hpp"
#include <iostream>
#include <numeric>

NeuralNetwork_fit::NeuralNetwork_fit(
    NeuralNetwork_impl::NetworkData _input,
    NeuralNetwork_impl::NetworkData _desiredOutput, int epochs, int _lossStep,
    NeuralNetwork_impl *impl)
    : lossStep(_lossStep), actualEpoch(), epochs(epochs), input(_input),
      desiredOutput(_desiredOutput), net_impl(impl) {}

void NeuralNetwork_fit::fit() {
      while (actualEpoch++ < epochs) {
            auto batch = prepareEpoch(input, desiredOutput);
            auto output = net_impl->GenerateOutput(batch->input, batch->output);
            net_impl->RecalculateWeights();
            calculeLoss(output, batch->output, epochs);
      }
}

std::unique_ptr<NeuralNetwork_fit::Batch>
NeuralNetwork_fit::prepareEpoch(NeuralNetwork_impl::NetworkData inputData,
                                NeuralNetwork_impl::NetworkData outputData) {
      static long unsigned int indexIn = 0, indexOut = 0;
      auto batch = std::make_unique<NeuralNetwork_fit::Batch>(
          net_impl->getInputSize(), net_impl->getOutputSize());
      std::uninitialized_copy(inputData.begin() + indexIn,
                              inputData.begin() + indexIn +
                                  net_impl->getInputSize(),
                              batch->input.begin());
      std::uninitialized_copy(outputData.begin() + indexOut,
                              outputData.begin() + net_impl->getOutputSize() +
                                  indexOut,
                              batch->output.begin());
      if (indexIn + net_impl->getInputSize() < inputData.size())
            indexIn += net_impl->getInputSize();
      else
            indexIn = 0;
      if (indexOut + net_impl->getOutputSize() < outputData.size())
            indexOut += net_impl->getOutputSize();
      else
            indexOut = 0;
      return batch;
}
double NeuralNetwork_fit::calculeLoss(std::vector<double> desiredOut,
                                      NeuralNetwork_impl::NetworkData out,
                                      int epochs) {
      static int batch_ind = 0;
      static std::vector<double> batchLoss{};
      int n = out.size();
      double loss = 0;
      for (int i = 0; i < n; i++) {
            loss += std::abs(out[i] - desiredOut[i]);
      }
      loss /= n;
      if (batch_ind++ == lossStep) {
            batch_ind = 0;
            showLoss(batchLoss, epochs);
            batchLoss.clear();
      } else
            batchLoss.push_back(loss);
      return loss;
}

void NeuralNetwork_fit::showLoss(std::vector<double> batchLoss, int epochs) {
      int div = epochs / 10;
      double _loss =
          std::accumulate(batchLoss.begin(), batchLoss.end(), 0.0f) / lossStep;
      if ((actualEpoch % div) == 0) {
            for (int i = 0; i < _loss * 10; i++) {
                  std::cout << "█";
            }
            std::cout << " " << _loss << " in epoch: " << actualEpoch << " \n";
      }
}
