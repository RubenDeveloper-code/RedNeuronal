#ifndef __NEURON_HPP__
#define __NEURON_HPP__

#include "Algorithm.hpp"
#include "NeuronActivation.hpp"
#include <memory>
#include <vector>
struct Connection;
class Neuron {

    public:
      using Connections = std::vector<std::shared_ptr<Connection>>;
      using Neurons = std::vector<Neuron>;
      enum class TYPE { INPUT, WIDE, OUTPUT };
      Neuron(std::shared_ptr<NeuronActivations::activation> act,
             std::shared_ptr<Algorithms::OptimizationAlgorithm> opt, TYPE type);
      inline void setValue(int _y) {
            if (type == TYPE::INPUT)
                  y = _y;
            else if (type == TYPE::OUTPUT)
                  targetValue = _y;
      }
      void makeConnections(Neurons &target, int prevLayerSize);
      double calculateValue();
      void fixInputWeights();
      void checkError();
      // debuig
      double y;

    private:
      std::shared_ptr<NeuronActivations::activation> activation;
      std::shared_ptr<Algorithms::OptimizationAlgorithm> optimizationAlgorithm;
      TYPE type;
      Connections prevConnections;
      Connections nextConnections;
      double error;
      double bias = 1.0;
      // algoritmo de pesos;
      double alpha = 0.000001;
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
