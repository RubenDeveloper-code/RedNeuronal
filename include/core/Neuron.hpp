#ifndef __NEURON_HPP__
#define __NEURON_HPP__

#include "../algorithms/Activations.hpp"
#include "../algorithms/LossFuctions.hpp"
#include "../algorithms/Optimizers.hpp"
#include "../data/Data.hpp"
#include <algorithm>
#include <memory>
#include <utility>
#include <vector>
struct Connection;
class Neuron {

    public:
      using Connections = std::vector<std::shared_ptr<Connection>>;
      using Neurons = std::vector<Neuron>;
      enum class TYPE { INPUT, HIDE, OUTPUT };

      Neuron(std::shared_ptr<Activations::activation> act,
             std::shared_ptr<Optimizers::Optimizer> opt,
             std::shared_ptr<LossFuctions::LossFunction> lossFoo, TYPE type);

      void makeConnections(Neurons &target, int prevLayerSize);
      double calculateValue();
      void recomputeParameters(std::vector<double> activations = {},
                               std::vector<double> targets = {});
      long double computeGradient(double, int, std::vector<double>,
                                  std::vector<double>);
      inline void setValue(int _y) {
            if (type == TYPE::INPUT)
                  y = _y;
            else if (type == TYPE::OUTPUT)
                  targetValue = _y;
      }
      TYPE type;
      // debuig
      double y;

    private:
      std::shared_ptr<Activations::activation> activation;
      std::shared_ptr<Optimizers::Optimizer> optimizationAlgorithm;
      std::shared_ptr<LossFuctions::LossFunction> lossFunction;
      Connections prevConnections;
      Connections nextConnections;
      double delta;
      double bias = 1.0;
      double prevY;

      double targetValue;
};

struct Connection {
      Connection(Neuron &_targetNeuron, std::shared_ptr<double> _weight)
          : targetNeuron{_targetNeuron}, weight{_weight} {};
      Neuron &targetNeuron;
      std::shared_ptr<double> weight;
};
#endif
