#ifndef __NETWORK_ALGORITHS_HPP__
#define __NETWORK_ALGORITHS_HPP__

#include "NetworkGlobalResources.hpp"
#include <iostream>
namespace NetworkAlgorithms {
// algritmos diovididos por funcion, E/ unos modifican alpha y eso
// se inicializara con un algoritmo generuco y sus valores, y el valro que
// modifica, para llamarlo una funcion de algorithm alha ejecutara los
// algoritmos en true

struct Algorithm {
      virtual void init(GlobalResourses *globalResourses) = 0;
      virtual void run() = 0;
      virtual ~Algorithm() = default;
};
struct WarmUp : public Algorithm {
      WarmUp(){};
      WarmUp(double _initialAlpha, double _finalAlpha, int _limitEpochs)
          : initialAlpha(_initialAlpha), finalAlpha(_finalAlpha),
            limitEpochs(_limitEpochs){};
      void init(GlobalResourses *res) override {
            globalResourses = res;
            isInit = true;
      }
      void run() override {
            if (isInit) {
                  if (*globalResourses->epochs_it <= limitEpochs)
                        *globalResourses->alpha =
                            initialAlpha + (*globalResourses->epochs_it /
                                            static_cast<float>(limitEpochs)) *
                                               (finalAlpha - initialAlpha);
                  else
                        *globalResourses->alpha = finalAlpha;
            }
      }
      GlobalResourses *globalResourses;
      double initialAlpha;
      double finalAlpha;
      int limitEpochs;
      bool isInit = false;
};

struct DecayLearningRate : public Algorithm {
      DecayLearningRate(){};
      DecayLearningRate(double _initialAlpha, double _finalAlpha,
                        int _limitEpochs)
          : initialAlpha(_initialAlpha), finalAlpha(_finalAlpha),
            limitEpochs(_limitEpochs){};
      void init(GlobalResourses *res) override {
            globalResourses = res;
            isInit = true;
      }
      void run() override {
            if (isInit) {
                  if (*globalResourses->epochs_it <= limitEpochs)
                        *globalResourses->alpha =
                            initialAlpha - (*globalResourses->epochs_it /
                                            static_cast<float>(limitEpochs)) *
                                               (finalAlpha - initialAlpha);
                  else
                        *globalResourses->alpha = finalAlpha;
            }
      }
      GlobalResourses *globalResourses;
      double initialAlpha;
      double finalAlpha;
      int limitEpochs;
      bool isInit = false;
};

struct AlgorithmsAlpha {
    private:
      enum class ALPHAALGORITHMS { WARMUP, DECAYLEARNINGRATE, UNDEFINED };
      WarmUp warm_up;
      DecayLearningRate decayLearningRate;
      ALPHAALGORITHMS active = AlgorithmsAlpha::ALPHAALGORITHMS::UNDEFINED;

    public:
      void upWarmUp(WarmUp &&wu) {
            warm_up = WarmUp(wu);
            active = ALPHAALGORITHMS::WARMUP;
      }
      void upDecayLearningRate(DecayLearningRate &&dlr) {
            decayLearningRate = DecayLearningRate(dlr);
            active = ALPHAALGORITHMS::DECAYLEARNINGRATE;
      }

      void init(GlobalResourses *res) {
            switch (active) {
            case ALPHAALGORITHMS::WARMUP:
                  warm_up.init(res);
                  break;
            case ALPHAALGORITHMS::DECAYLEARNINGRATE:
                  decayLearningRate.init(res);
                  break;
            case ALPHAALGORITHMS::UNDEFINED:
                  break;
            }
      }
      void run() {
            switch (active) {
            case ALPHAALGORITHMS::WARMUP:
                  warm_up.run();
                  break;
            case ALPHAALGORITHMS::DECAYLEARNINGRATE:
                  decayLearningRate.run();
                  break;
            case ALPHAALGORITHMS::UNDEFINED:
                  break;
            }
      }
};
} // namespace NetworkAlgorithms

#endif
