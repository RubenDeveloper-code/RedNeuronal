#include "NeuralNetwork.hpp"
#include "NeuralNetwork_impl.hpp"
#include <iostream>
#include <memory>
#include <ostream>

NeuralNetwork::NeuralNetwork(NetworkDimentions netDims, Neuron::Activation act)
    : net_impl(netDims, act) {}

void NeuralNetwork::fit(NeuralNetwork_impl::NetworkData inputData,
                        NeuralNetwork_impl::NetworkData outputData,
                        int epochs) {
      int indexIn = 0, indexOut = 0;
      while (epochs-- > 0) {
            NeuralNetwork_impl::NetworkData input(net_impl.getInputSize()),
                output(net_impl.getOutputSize());
            // 1 2 3 4 5
            std::uninitialized_copy(inputData.begin() + indexIn,
                                    inputData.begin() + indexIn +
                                        net_impl.getInputSize(),
                                    input.begin());
            std::uninitialized_copy(outputData.begin() + indexOut,
                                    outputData.begin() +
                                        net_impl.getOutputSize() + indexOut,
                                    output.begin());
            auto Output = net_impl.GenerateOutput(input, output);
            std::cout << "Entradas" << std::endl;
            for (auto in = input.begin(); in != input.end(); in++)
                  std::cout << *in << std::endl;

            std::cout << "Salidas" << std::endl;
            for (auto out : Output) {
                  std::cout << out << std::endl;
            }
            net_impl.RecalculateWeights();
            if (indexIn + net_impl.getInputSize() < inputData.size())
                  indexIn += net_impl.getInputSize();
            else
                  indexIn = 0;
            if (indexOut + net_impl.getOutputSize() < outputData.size())
                  indexOut += net_impl.getOutputSize();
            else
                  indexOut = 0;
      }
}
