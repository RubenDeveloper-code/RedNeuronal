#include "Neuron.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>

double randomReal(double li, double ls) {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<> dis(li, ls);
      return dis(gen);
}

Neuron::Neuron(double (*_activation)(double), TYPE _type)
    : activation{_activation}, type{_type} {}

void Neuron::makeConnections(Neurons &target, int prevLayerSize, TYPE type) {
      // se inicializan a partir de las input enurons de la capa
      float std_dev;
      if (type == TYPE::WIDE)
            std_dev = std::sqrt(2.0 / (double)prevLayerSize);
      else
            std_dev = std::sqrt(1.0 / (double)prevLayerSize);

      // Inicializar el generador de números aleatorios
      std::random_device rd;
      std::mt19937 gen(rd());
      std::normal_distribution<float> dist(0.0, std_dev);
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
      y = activation(y);
      if (type == TYPE::OUTPUT) {
            return y;
      }
      return -1;
}
// revisar operaciones
void Neuron::checkError() {
      error = 0;
      if (type == TYPE::OUTPUT) {
            error = (y * (1.0 - y)) * (y - targetValue);
      } else if (type == TYPE::WIDE) {
            double summNextErrors = 0.0;
            for (const auto &nextConn : nextConnections) {
                  summNextErrors +=
                      *nextConn->weight * nextConn->targetNeuron.error;
            }
            int fooy = (prevY > 0) ? 1 : 0;
            error = fooy * summNextErrors;
      }
}

void Neuron::fixInputWeights() {
      checkError();
      bias -= alpha * error * 1.0;
      if (type == TYPE::INPUT)
            return;
      for (auto &prevConn : prevConnections) {
            *prevConn->weight -= alpha * error * prevConn->targetNeuron.y;
      }
}
