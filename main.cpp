#include "include/Model.hpp"
#include "include/algorithms/Activations.hpp"
#include "include/algorithms/Optimizers.hpp"
#include "include/data/DataSetProcess.hpp"
#include "include/data/Normalizers.hpp"
#include "include/designs/LayerDesign.hpp"
#include "include/designs/ModelDesign.hpp"
#include "include/designs/train/AlgorithmsSpects.hpp"
#include "include/designs/train/EarlyStopSpects.hpp"
#include "include/designs/train/TrainSpects.hpp"
#include "include/network/body/Layer.hpp"
#include "include/types/data/TrainingDataSet.hpp"
#include <iostream>
#include <memory>

#define BEST_CKPT "../temp/temp_best.ckpt"

int main() {
      Model model(ModelDesign::LossFuction::MEAN_SQUARED_ERROR);
      model.addLayer(LayerDesign{LayerDesign::LayerClass::INPUT,
                                 LayerDesign::Activations::RELU,
                                 LayerDesign::Optimizers::ADAMS, 12, 0});
      model.addLayer(LayerDesign{LayerDesign::LayerClass::HIDE,
                                 LayerDesign::Activations::RELU,
                                 LayerDesign::Optimizers::ADAMS, 32, 0});
      model.addLayer(LayerDesign{LayerDesign::LayerClass::HIDE,
                                 LayerDesign::Activations::RELU,
                                 LayerDesign::Optimizers::ADAMS, 16, 0});
      model.addLayer(LayerDesign{LayerDesign::LayerClass::OUTPUT,
                                 LayerDesign::Activations::REGRESSION,
                                 LayerDesign::Optimizers::ADAMS, 1});
      // son la misma mamada nomas cambia los argumentos
      // model.upAlphaAlgoritm(AlgorithmsSpects::AlphaModifier::DECAY,
      //   {1, 0.1, 10, true, 0.001});
      // model.upEarlyStop({5, true});
      DataSetProcess dataset("../res/Student_performance_data _.csv");
      dataset.applyNormalization(Normalizations::TYPE::ZCORE);
      std::vector<std::string> tagsInput = {"Age",
                                            "Gender",
                                            "Ethnicity",
                                            "ParentalEducation",
                                            "StudyTimeWeekly",
                                            "Absences",
                                            "Tutoring",
                                            "ParentalSupport",
                                            "Extracurricular",
                                            "Sports",
                                            "Music",
                                            "Volunteering"};
      std::vector<std::string> tagsOutput = {"GradeClass"};
      TrainingDataSet training_dataset =
          dataset.getTrainingDataSet(tagsInput, tagsOutput, 70, 20, 10);

      auto test = training_dataset.test_dataset;
      TrainSpects train_spects{
          std::move(training_dataset),
          60000,
          1,
          10e-3,
          0.000001,
          10000,
          "../checkpoints",
      };
      model.fit(train_spects, "../temp/temp_best.ckpt");
      // model.loadCheckpoint(BEST_CKPT);
      auto out = model.predict((test.begin() + 20)->input);
      std::cout << "->debe ser: " << (test.begin() + 20)->output[0]
                << " es: " << out[0] << std::endl;
      auto out2 = model.predict(
          {17, 0, 3, 2, 18.397418682610855, 10, 0, 4, 0, 0, 0, 0});
      std::cout << "->debe ser 3.0 " << out2[0] << std::endl;
}
