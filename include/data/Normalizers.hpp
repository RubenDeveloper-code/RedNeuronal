#ifndef __NORMALIZATIONS_HPP__
#define __NORMALIZATIONS_HPP__

#include "DataSet.hpp"
#include <cmath>
#include <functional>
#include <memory>
#include <vector>

#define OUT 0
#define IN 1
namespace Normalizations {
enum class TYPE { ZCORE };

std::vector<std::reference_wrapper<double>> getColumnRefs(DataSet &data,
                                                          int element, int id);
void computeDataToContainer(std::vector<double> &container, DataSet &data,
                            double (*ecuation)(double, double),
                            double (*finalOp)(double, double),
                            std::vector<double> means = {});
std::vector<double> computeMean(DataSet &data);
std::vector<double> computeStandartDeviation(DataSet &data,
                                             std::vector<double> means);
struct Normalization {
      virtual DataSet normalizeData(DataSet &data) = 0;
      virtual double revertNormalization(double val, double column_id) = 0;
      virtual double individualNormalization(double val, double column_id) = 0;

      virtual ~Normalization(){};
};
struct Zscore : public Normalization {
      DataSet normalizeData(DataSet &data) override {
            means = computeMean(data);
            standartDeviations = computeStandartDeviation(data, means);
            for (auto field = 0; field < data.set[0].input.size(); field++) {
                  auto irefs = getColumnRefs(data, field, IN);
                  for (auto &ref : irefs) {
                        ref.get() = (ref.get() - means[field]) /
                                    standartDeviations[field];
                  }
            }
            for (auto field = 0; field < data.set[0].output.size(); field++) {
                  auto irefs = getColumnRefs(data, field, OUT);
                  for (auto &ref : irefs) {
                        ref.get() =
                            (ref.get() -
                             means[field + data.set[0].input.size()]) /
                            standartDeviations[field +
                                               data.set[0].input.size()];
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

std::unique_ptr<Normalization> newInstance(TYPE type);

} // namespace Normalizations

#endif
