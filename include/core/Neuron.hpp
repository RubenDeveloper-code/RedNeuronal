#ifndef __NEURON_HPP__
#define __NEURON_HPP__

#include "../../include/types/Parameters.hpp"
#include "../algorithms/Activations.hpp"
#include "../algorithms/LossFuctions.hpp"
#include "../algorithms/Optimizers.hpp"
#include <memory>
#include <vector>
struct Connection;
class Neuron {
    public:
      enum class TYPE { INPUT, HIDE, OUTPUT };
      using Connections = std::vector<std::shared_ptr<Connection>>;
      using Neurons = std::vector<Neuron>;
      Neuron(std::shared_ptr<Activations::activation> act,
             std::shared_ptr<Optimizers::Optimizer> opt,
             std::shared_ptr<LossFuctions::LossFunction> lossFoo, TYPE type);
      void makeConnections(Neurons &target, int prevLayerSize);

      double computeActivation();
      void recomputeParameters(std::vector<double> activations = {},
                               std::vector<double> targets = {});
      long double computeGradient(double, int, std::vector<double>,
                                  std::vector<double>);
      void initializeAccordingType(int _y);
      Parameters getParameters();
      void loadParameters(Parameters parameters);
      TYPE type;

    private:
      double neuron_activation;
      std::shared_ptr<Activations::activation> activation;
      std::shared_ptr<Optimizers::Optimizer> optimizationAlgorithm;
      std::shared_ptr<LossFuctions::LossFunction> lossFunction;
      Connections prevConnections;
      Connections nextConnections;
      double delta;
      double bias = 1.0;
      double weighted_sum;
      double targetValue;
};
struct Connection {
      Connection(Neuron &_targetNeuron, std::shared_ptr<double> _weight)
          : targetNeuron{_targetNeuron}, weight{_weight} {};
      Neuron &targetNeuron;
      std::shared_ptr<double> weight;
};
#endif
