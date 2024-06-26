#include "../include/Neuron.hpp"
#include "../include/NeuronActivation.hpp"
#include "../include/OptimizationAlgorithms.hpp"
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <vector>
using namespace std;

#define WEIGHT 1
#define BIAS 0
// ya lestoy agarrando el pedo

Neuron::Neuron(
    std::shared_ptr<NeuronActivations::activation> _activation,
    std::shared_ptr<OptimizationAlgorithms::OptimizationAlgorithm> opt,
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

double Neuron::calculateValue() {
      y = 0;
      for (const auto &prevConn : prevConnections) {
            y += (prevConn->targetNeuron.y * *prevConn->weight);
      }
      y += (bias);
      prevY = y;
      y = activation->function(y);
      if (type == TYPE::OUTPUT) {
            return y;
      }
      return -1;
}
long double
Neuron::computeGradient(double prevActivation, int theta,
                        std::vector<OutputNetworkData> activations,
                        std::vector<OutputNetworkData> targetValues) {
      const int N = activations.size();
      if (type == TYPE::OUTPUT) {
            double act = activation->derivative(prevY, y);
            double accomulate_loss{};
            for (auto it = 0; it < N; it++) {
                  accomulate_loss += lossFunction->derivative(activations[it],
                                                              targetValues[it]);
            }
            delta = (accomulate_loss / N) * act;
            long double gradient = delta * -prevActivation;
            return gradient;
      } else if (type == TYPE::WIDE) {
            double summ_deltas{};
            for (auto &conn : nextConnections) {
                  summ_deltas += (*conn->weight * conn->targetNeuron.delta);
            }
            summ_deltas *= activation->derivative(prevY, y);
            delta = summ_deltas;
            long double gradient = delta * -prevActivation;
            return gradient;
      }
      return 0;
}

void Neuron::recomputeParameters(
    std::vector<OutputNetworkData> minibatch_activations,
    std::vector<OutputNetworkData> minibatch_targets) {
      if (type == TYPE::INPUT)
            return;
      for (auto &prevConn : prevConnections) {
            *prevConn->weight = optimizationAlgorithm->optimizeWeigth(
                {*prevConn->weight,
                 computeGradient(prevConn->targetNeuron.y, WEIGHT,
                                 minibatch_activations, minibatch_targets)});
      }
      bias = optimizationAlgorithm->optimizeBias(
          {bias, computeGradient(1.0, BIAS, minibatch_activations,
                                 minibatch_targets)});
}
