#ifndef __NEURALNETWORK_FIT_HPP__
#define __NEURALNETWORK_FIT_HPP__

#include "NeuralNetwork_impl.hpp"
#include <vector>
class NeuralNetwork_fit {
    public:
      struct Batch {
            Batch(int inSize, int outSize) : input(inSize), output(outSize) {}
            NeuralNetwork_impl::NetworkData input;
            NeuralNetwork_impl::NetworkData output;
      };
      NeuralNetwork_fit(NeuralNetwork_impl::NetworkData input,
                        NeuralNetwork_impl::NetworkData desiredOutput,
                        int epochs, int lossStep, NeuralNetwork_impl *impl);
      void fit();

    private:
      int lossStep;
      int actualEpoch, epochs;
      NeuralNetwork_impl::NetworkData input;
      NeuralNetwork_impl::NetworkData desiredOutput;
      NeuralNetwork_impl *net_impl;
      std::unique_ptr<Batch>
      prepareEpoch(NeuralNetwork_impl::NetworkData input,
                   NeuralNetwork_impl::NetworkData output);
      double calculeLoss(std::vector<double> out,
                         NeuralNetwork_impl::NetworkData desiredOut,
                         int epochs);
      void showLoss(std::vector<double> batchLoss, int epochs);
};

#endif
