#include "../../include/data/Normalizers.hpp"

void Normalizations::computeDataToContainer(std::vector<double> &container,
                                            DataSet &data,
                                            double (*ecuation)(double, double),
                                            double (*finalOp)(double, double),
                                            std::vector<double> means) {
      const int N = data.size();
      for (auto itElementInput = 0; itElementInput < data.set[0].input.size();
           itElementInput++) {
            double summ = 0;
            auto refs = getColumnRefs(data, itElementInput, IN);
            for (auto &ref : refs) {
                  if (means.empty())
                        summ += ecuation(ref.get(), 0);
                  else
                        summ += ecuation(ref.get(), means[itElementInput]);
            }
            container.push_back(finalOp(summ, N));
      }
      for (auto itElementOutput = 0;
           itElementOutput < data.set[0].output.size(); itElementOutput++) {
            double summ = 0;
            auto refs = getColumnRefs(data, itElementOutput, OUT);
            for (auto &ref : refs) {
                  if (means.empty())
                        summ += ecuation(ref.get(), 0);
                  else
                        summ += ecuation(ref.get(),
                                         means[itElementOutput +
                                               (data.set[0].input.size())]);
            }
            container.push_back(finalOp(summ, N));
      }
}
std::vector<std::reference_wrapper<double>>
Normalizations::getColumnRefs(DataSet &data, int element, int id) {
      std::vector<std::reference_wrapper<double>> refs;
      for (auto itData = 0; itData < data.size(); itData++) {
            if (id == IN)
                  refs.push_back(data.set[itData].input[element]);
            else
                  refs.push_back(data.set[itData].output[element]);
      }
      return refs;
}
std::vector<double> Normalizations::computeMean(DataSet &data) {
      std::vector<double> means;
      computeDataToContainer(
          means, data, [](double x, double) { return x; },
          [](double summ, double N) { return summ / N; });
      return means;
}

std::vector<double>
Normalizations::computeStandartDeviation(DataSet &data,
                                         std::vector<double> means) {
      std::vector<double> standartDeviations;
      computeDataToContainer(
          standartDeviations, data,
          [](double x, double mean) { return pow(x - mean, 2); },
          [](double summ, double N) { return sqrt(summ / N); }, means);
      return standartDeviations;
}
std::unique_ptr<Normalizations::Normalization>
Normalizations::newInstance(TYPE type) {
      switch (type) {
      case TYPE::ZCORE:
            return std::make_unique<Zscore>();
      }
      return std::make_unique<Zscore>();
}
