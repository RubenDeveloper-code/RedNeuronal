#include "../../include/core/Neuron.hpp"
#include "../../include/algorithms/Activations.hpp"
#include "../../include/algorithms/Optimizers.hpp"
#include "../../include/types/Parameters.hpp"
#include <iostream>
#include <memory>
#include <random>
#include <vector>
using namespace std;

#define WEIGHT 1
#define BIAS 0

Neuron::Neuron(std::shared_ptr<Activations::activation> _activation,
               std::shared_ptr<Optimizers::Optimizer> opt,
               std::shared_ptr<LossFuctions::LossFunction> lossFoo, TYPE _type)
    : activation{_activation}, optimizationAlgorithm{opt},
      lossFunction(lossFoo), type{_type} {}

void Neuron::makeConnections(Neurons &target, int prevLayerSize) {
      double std_dev = activation->getDevStandart(prevLayerSize);
      std::random_device rd;
      std::mt19937 gen(rd());
      std::normal_distribution<double> dist(0.0, std_dev);

      for (auto &targetNeuron : target) {
            std::shared_ptr<double> shared_weight{
                std::make_shared<double>(dist(gen))};
            std::shared_ptr<Connection> next =
                std::make_shared<Connection>(targetNeuron, shared_weight);
            nextConnections.push_back(next);
            std::shared_ptr<Connection> prev =
                std::make_shared<Connection>(*this, shared_weight);
            targetNeuron.prevConnections.push_back(prev);
      }
}

double Neuron::computeActivation() {
      weighted_sum = 0;
      for (auto &prevConn : prevConnections) {
            weighted_sum +=
                (prevConn->targetNeuron.neuron_activation * *prevConn->weight);
      }
      weighted_sum += bias;
      neuron_activation = activation->function(weighted_sum);
      if (type == TYPE::OUTPUT) {
            return neuron_activation;
      }
      return -1;
}
long double
Neuron::computeGradient(double prevActivation, int theta,
                        std::vector<double> activations_in_minibatch,
                        std::vector<double> targets_in_minibatch) {
      if (type == TYPE::OUTPUT) {
            double act =
                activation->derivative(weighted_sum, neuron_activation);
            double loss{};
            loss = lossFunction->derivative(activations_in_minibatch,
                                            targets_in_minibatch);
            delta = loss * act;
            long double gradient = delta * -prevActivation;
            return gradient;
      } else if (type == TYPE::HIDE) {
            double summ_deltas{};
            for (auto &conn : nextConnections) {
                  summ_deltas += (*conn->weight * conn->targetNeuron.delta);
            }
            summ_deltas *=
                activation->derivative(weighted_sum, neuron_activation);
            delta = summ_deltas;
            long double gradient = delta * -prevActivation;
            return gradient;
      }
      return 0;
}

void Neuron::recomputeParameters(std::vector<double> activations_in_minibatch,
                                 std::vector<double> targets_in_minibatch) {
      if (type == TYPE::INPUT)
            return;
      for (auto &prevConn : prevConnections) {
            *prevConn->weight = optimizationAlgorithm->optimizeWeigth(
                {*prevConn->weight,
                 computeGradient(prevConn->targetNeuron.neuron_activation,
                                 WEIGHT, activations_in_minibatch,
                                 targets_in_minibatch)});
      }
      bias = optimizationAlgorithm->optimizeBias(
          {bias, computeGradient(1.0, BIAS, activations_in_minibatch,
                                 targets_in_minibatch)});
}
Parameters Neuron::getParameters() {
      std::vector<double> weights;
      for (auto &prevConn : prevConnections) {
            weights.push_back(*prevConn->weight);
      }
      return Parameters{weights, bias};
}

void Neuron::loadParameters(Parameters parameters) {
      for (auto w_pos = 0; w_pos < prevConnections.size(); w_pos++) {
            *prevConnections[w_pos]->weight = parameters.weights[w_pos];
      }
      bias = parameters.bias;
}

void Neuron::initializeAccordingType(int init_val) {
      if (type == TYPE::INPUT)
            neuron_activation = init_val;
      else if (type == TYPE::OUTPUT)
            targetValue = init_val;
}
