#ifndef __NEURALNETWORKSETTERDATA_HPP__
#define __NEURALNETWORKSETTERDATA_HPP__
#include "Data.hpp"
#include "Layer.hpp"
#include "Neuron.hpp"
#include <algorithm>
#include <vector>

struct SetterData {
      SetterData() {}
      SetterData(NetworkTrainData dataTrain) : trainData(dataTrain) {
            itData = trainData.begin();
      }
      void preparePrediction(Layer *input, InputNetworkData dataInput) {
            int inputIt{};
            std::for_each(dataInput.begin(), dataInput.end(),
                          [&inputIt, &input](double value) {
                                input->neurons[inputIt++].setValue(value);
                          });
      }
      Data prepareNextEpoch(Layer *input, Layer *output) {
            Data data = *itData;
            int inputIt{};
            std::for_each(data.input.begin(), data.input.end(),
                          [&inputIt, &input](double value) {
                                input->neurons[inputIt++].setValue(value);
                          });
            int outputIt{};
            std::for_each(data.output.begin(), data.output.end(),
                          [&outputIt, &output](double value) {
                                output->neurons[outputIt++].setValue(value);
                          });

            if ((++itData) == trainData.end())
                  itData = trainData.begin();
            return data;
      }
      double getDataSize() { return trainData.size(); }
      OutputNetworkData getTarget() { return itData->output; }
      InputNetworkData getInput() { return itData->input; }

    private:
      NetworkTrainData::iterator itData;
      NetworkTrainData trainData;
};

#endif
