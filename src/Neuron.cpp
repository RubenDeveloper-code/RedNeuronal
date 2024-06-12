#include "../include/Neuron.hpp"
#include "../include/Algorithm.hpp"
#include "../include/NeuronActivation.hpp"
#include <memory>
#include <random>

double randomReal(double li, double ls) {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<> dis(li, ls);
      return dis(gen);
}

Neuron::Neuron(std::shared_ptr<NeuronActivations::activation> _activation,
               std::shared_ptr<Algorithms::OptimizationAlgorithm> opt,
               TYPE _type)
    : activation{_activation}, optimizationAlgorithm{opt}, type{_type} {}

void Neuron::makeConnections(Neurons &target, int prevLayerSize) {
      double std_dev = activation->getDevStandart(prevLayerSize);
      std::random_device rd;
      std::mt19937 gen(rd());

      std::normal_distribution<double> dist(0.0, std_dev);
      for (auto &targetNeuron : target) {
            double w = dist(gen);
            std::shared_ptr<double> shared_weight{std::make_shared<double>(w)};
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

void Neuron::checkError() {
      error = 0;
      if (type == TYPE::OUTPUT) {
            error = activation->derivative(prevY, y) * (y - targetValue);
      } else if (type == TYPE::WIDE) {
            double summNextErrors = 0.0;
            for (const auto &nextConn : nextConnections) {
                  summNextErrors +=
                      *nextConn->weight * nextConn->targetNeuron.error;
            }
            error = activation->derivative(prevY, y) * summNextErrors;
      }
}

// debe ser usado el -=
void Neuron::fixInputWeights() {
      checkError();
      bias -= alpha * error * 1.0;
      if (type == TYPE::INPUT)
            return;
      for (auto &prevConn : prevConnections) {
            *prevConn->weight -= alpha * error * prevConn->targetNeuron.y;
      }
}
