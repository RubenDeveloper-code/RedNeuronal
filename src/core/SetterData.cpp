#include "../../include/core/SetterData.hpp"
#include <algorithm>
#include <chrono>
#include <random>
#include <vector>

SetterData::SetterData(DataSet _dataset) : dataset(_dataset) {
      dataset.initializeIterator();
}
void SetterData::preparePrediction(Layer &input, InputNetworkData dataInput) {
      int inputIt{};
      std::for_each(dataInput.begin(), dataInput.end(),
                    [&inputIt, &input](double value) {
                          input.neurons[inputIt++].setValue(value);
                    });
}
Data SetterData::prepareNextEpoch(Layer &input, Layer &output) {
      Data data = *dataset.iterator;
      int inputIt{};
      std::for_each(data.input.begin(), data.input.end(),
                    [&inputIt, &input](double value) {
                          input.neurons[inputIt++].setValue(value);
                    });
      int outputIt{};
      std::for_each(data.output.begin(), data.output.end(),
                    [&outputIt, &output](double value) {
                          output.neurons[outputIt++].setValue(value);
                    });

      if ((++dataset.iterator) == dataset.end()) {
            unsigned seed =
                std::chrono::system_clock::now().time_since_epoch().count();
            std::default_random_engine engine(seed);
            std::shuffle(dataset.begin(), dataset.end(), engine);
            dataset.iterator = dataset.begin();
      }
      return data;
}
