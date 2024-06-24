#ifndef __NEURON_HPP__
#define __NEURON_HPP__

#include "Algorithm.hpp"
#include "Data.hpp"
#include "LossFuctions.hpp"
#include "NeuronActivation.hpp"
#include <algorithm>
#include <memory>
#include <utility>
#include <vector>
struct Connection;
class Neuron {

    public:
      using Connections = std::vector<std::shared_ptr<Connection>>;
      using Neurons = std::vector<Neuron>;
      enum class TYPE { INPUT, WIDE, OUTPUT };

      Neuron(std::shared_ptr<NeuronActivations::activation> act,
             std::shared_ptr<Algorithms::OptimizationAlgorithm> opt,
             std::shared_ptr<LossFuctions::LossFunction> lossFoo, TYPE type);

      void makeConnections(Neurons &target, int prevLayerSize);
      double calculateValue();
      void recomputeParameters(std::vector<OutputNetworkData>,
                               std::vector<OutputNetworkData>);
      long double computeGradient(double, int, std::vector<OutputNetworkData>,
                                  std::vector<OutputNetworkData>);
      inline void setValue(int _y) {
            if (type == TYPE::INPUT)
                  y = _y;
            else if (type == TYPE::OUTPUT)
                  targetValue = _y;
      }

      // debuig
      double y;

    private:
      TYPE type;
      std::shared_ptr<NeuronActivations::activation> activation;
      std::shared_ptr<Algorithms::OptimizationAlgorithm> optimizationAlgorithm;
      std::shared_ptr<LossFuctions::LossFunction> lossFunction;
      Connections prevConnections;
      Connections nextConnections;
      double delta;
      double bias = 1.0;
      double alpha = 0.00001;
      double prevY;

      double targetValue;
      long double weight_gradient;
      long double bias_gradient;
};

struct Connection {
      Connection(Neuron &_targetNeuron, std::shared_ptr<double> _weight)
          : targetNeuron{_targetNeuron}, weight{_weight} {};
      Neuron &targetNeuron;
      std::shared_ptr<double> weight;
};
#endif
