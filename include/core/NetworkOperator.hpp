#ifndef __NETWORK_OPERATOR_HPP__
#define __NETWORK_OPERATOR_HPP__

#include "../../include/types/Parameters.hpp"
#include "Network.hpp"
#include <vector>
class NetworkOperator {
    public:
      OutputNetworkData computeNetworkOutput(Network &network);
      void recalculateNetworkParameters(Network &network,
                                        std::vector<PairOutputs> pair_outputs);
      std::vector<Parameters> getNetworkParameters(Network &network);
      void loadCheckpointParameters(Network &networ,
                                    std::vector<Parameters> network_params);

    private:
      std::vector<PairOutputs>
      distributeResponsibilities(std::vector<PairOutputs> pair_outputs,
                                 int size_output);
};

#endif
