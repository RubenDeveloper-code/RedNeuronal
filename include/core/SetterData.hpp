#ifndef __NEURALNETWORKSETTERDATA_HPP__
#define __NEURALNETWORKSETTERDATA_HPP__
#include "../data/Data.hpp"
#include "../data/DataSet.hpp"
#include "Layer.hpp"
struct SetterData {
      SetterData() {}
      SetterData(DataSet dataset);
      void preparePrediction(Layer &input, InputNetworkData dataInput);
      Data prepareNextEpoch(Layer &input, Layer &output);
      double getDataSize() { return dataset.size(); }

    private:
      DataSet dataset;
};
#endif
