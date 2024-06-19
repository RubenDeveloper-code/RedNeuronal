#include "../include/Neuron.hpp"
#include "../include/Algorithm.hpp"
#include "../include/NeuronActivation.hpp"
#include <memory>
#include <random>

Neuron::Neuron(std::shared_ptr<NeuronActivations::activation> _activation,
               std::shared_ptr<Algorithms::OptimizationAlgorithm> opt,
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

void Neuron::fixInputWeights() {
      if (type == TYPE::INPUT)
            return;
      for (auto &prevConn : prevConnections) {
            delta = checkError(prevConn->targetNeuron.y);
            *prevConn->weight = optimizationAlgorithm->optimizeWeigth(
                {*prevConn->weight, alpha, delta, prevConn->targetNeuron.y});
      }
      bias = optimizationAlgorithm->optimizeBias({bias, alpha, checkError(1)});
}

double Neuron::checkError(double prevActivation) {
      double error{};
      if (type == TYPE::OUTPUT) {
            error = activation->derivative(prevY, y) *
                    lossFunction->derivative(y, targetValue);
      } else if (type == TYPE::WIDE) {
            double summNextErrors = 0.0;
            for (const auto &nextConn : nextConnections) {
                  summNextErrors +=
                      *nextConn->weight * nextConn->targetNeuron.delta;
            }
            error = activation->derivative(prevY, y) * summNextErrors;
      }
      return error;
}
