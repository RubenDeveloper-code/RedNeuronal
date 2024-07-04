#ifndef __NORMALIZATIONS_HPP__
#define __NORMALIZATIONS_HPP__

#include "Data.hpp"
#include <cmath>
#include <functional>
#include <memory>
#include <vector>

#define OUT 0
#define IN 1

namespace Normalizations {
enum class TYPE { ZCORE };
struct Normalization {
      virtual NetworkTrainData normalizeData(NetworkTrainData &data) = 0;
      std::vector<std::reference_wrapper<double>>
      getColumnRefs(NetworkTrainData &data, int element, int id) {
            std::vector<std::reference_wrapper<double>> refs;
            for (auto itData = 0; itData < data.size(); itData++) {
                  if (id == IN)
                        refs.push_back(data[itData].input[element]);
                  else
                        refs.push_back(data[itData].output[element]);
            }
            return refs;
      }
      void computeDataToContainer(std::vector<double> &container,
                                  NetworkTrainData &data,
                                  double (*ecuation)(double, double),
                                  double (*finalOp)(double, double),
                                  std::vector<double> means = {}) {
            const int N = data.size();
            for (auto itElementInput = 0; itElementInput < data[0].input.size();
                 itElementInput++) {
                  double summ = 0;
                  auto refs = getColumnRefs(data, itElementInput, IN);
                  for (auto &ref : refs) {
                        if (means.empty())
                              summ += ecuation(ref.get(), 0);
                        else
                              summ +=
                                  ecuation(ref.get(), means[itElementInput]);
                  }
                  container.push_back(finalOp(summ, N));
            }
            for (auto itElementOutput = 0;
                 itElementOutput < data[0].output.size(); itElementOutput++) {
                  double summ = 0;
                  auto refs = getColumnRefs(data, itElementOutput, OUT);
                  for (auto &ref : refs) {
                        if (means.empty())
                              summ += ecuation(ref.get(), 0);
                        else
                              summ += ecuation(ref.get(),
                                               means[itElementOutput +
                                                     (data[0].input.size())]);
                  }
                  container.push_back(finalOp(summ, N));
            }
      }
      std::vector<double> computeMean(NetworkTrainData &data) {
            std::vector<double> means;
            computeDataToContainer(
                means, data, [](double x, double) { return x; },
                [](double summ, double N) { return summ / N; });
            return means;
      }
      std::vector<double> computeStandartDeviation(NetworkTrainData &data,
                                                   std::vector<double> means) {
            std::vector<double> standartDeviations;
            computeDataToContainer(
                standartDeviations, data,
                [](double x, double mean) { return pow(x - mean, 2); },
                [](double summ, double N) { return sqrt(summ / N); }, means);
            return standartDeviations;
      }
      virtual double revertNormalization(double val, double column_id) = 0;
      virtual double individualNormalization(double val, double column_id) = 0;
      virtual ~Normalization(){};
};
struct Zscore : public Normalization {
      NetworkTrainData normalizeData(NetworkTrainData &data) override {
            means = computeMean(data);
            standartDeviations = computeStandartDeviation(data, means);
            for (auto field = 0; field < data[0].input.size(); field++) {
                  auto irefs = getColumnRefs(data, field, IN);
                  for (auto &ref : irefs) {
                        ref.get() = (ref.get() - means[field]) /
                                    standartDeviations[field];
                  }
            }
            for (auto field = 0; field < data[0].output.size(); field++) {
                  auto irefs = getColumnRefs(data, field, OUT);
                  for (auto &ref : irefs) {
                        ref.get() =
                            (ref.get() - means[field + data[0].input.size()]) /
                            standartDeviations[field + data[0].input.size()];
                  }
            }
            return data;
      }
      double revertNormalization(double val, double column_id) override {
            return val * standartDeviations[column_id] + means[column_id];
      };
      double individualNormalization(double val, double column_id) override {
            return (val - means[column_id]) / standartDeviations[column_id];
      }

    private:
      std::vector<double> means;
      std::vector<double> standartDeviations;
};

inline std::unique_ptr<Normalization> newInstance(TYPE type) {
      switch (type) {
      case TYPE::ZCORE:
            return std::make_unique<Zscore>();
      }
      return std::make_unique<Zscore>();
}

} // namespace Normalizations

#endif
