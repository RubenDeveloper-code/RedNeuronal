#include "../../include/designs/ModelDesign.hpp"
#include "../../include/alerts/handler.hpp"
#include <algorithm>

bool ModelDesign::checkIntegrity() {
      if (loss_function == ModelDesign::LossFuction::UNDEFINED)
            Handler::terminalUserError({"loss function undefined"});
      return sortLayerDesign();
}

bool ModelDesign::sortLayerDesign() {
      std::sort(design.begin(), design.end(),
                [](const LayerDesign &lda, const LayerDesign &ldb) {
                      return lda.type < ldb.type;
                });
      if (design[0].type != LayerDesign::LayerClass::INPUT)
            Handler::terminalUserError({"no input layer"});
      if (design[design.size() - 1].type != LayerDesign::LayerClass::OUTPUT)
            Handler::terminalUserError({"no output layer"});
      return true;
}
