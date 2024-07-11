#include "include/Model.hpp"
#include "include/algorithms/Activations.hpp"
#include "include/algorithms/Optimizers.hpp"
#include "include/core/Layer.hpp"
#include "include/data/DataSetProcess.hpp"
#include "include/data/Normalizers.hpp"
#include "include/designs/LayerDesign.hpp"
#include "include/designs/ModelDesign.hpp"
#include "include/designs/TrainSpects.hpp"
#include "include/types/TrainingDataSet.hpp"
#include <iostream>
#include <memory>

int main() {
      Model model(ModelDesign::LossFuction::MEAN_SQUARED_ERROR);
      model.addLayer(LayerDesign{LayerDesign::LayerClass::INPUT,
                                 LayerDesign::Activations::RELU,
                                 LayerDesign::Optimizers::ADAMS, 12});
      model.addLayer(LayerDesign{LayerDesign::LayerClass::HIDE,
                                 LayerDesign::Activations::RELU,
                                 LayerDesign::Optimizers::ADAMS, 32});
      model.addLayer(LayerDesign{LayerDesign::LayerClass::HIDE,
                                 LayerDesign::Activations::RELU,
                                 LayerDesign::Optimizers::ADAMS, 16});
      model.addLayer(LayerDesign{LayerDesign::LayerClass::OUTPUT,
                                 LayerDesign::Activations::REGRESSION,
                                 LayerDesign::Optimizers::ADAMS, 1});
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

      TrainSpects train_spects{std::move(training_dataset),
                               800,
                               1,
                               10e-3,
                               0.1,
                               10,
                               "../checkpoints",
                               10,
                               true};
      model.fit(train_spects);
      // model.loadCheckpoint("../checkpoints/checkpoint_2024-7-9_17_4_59.ckpt");
      auto out =
          model.predict({17, 1, 0, 2, 19.833722807854713, 7, 1, 2, 0, 0, 1, 0});
      std::cout << "->debe ser 2.0 " << out[0] << std::endl;
      auto out2 = model.predict(
          {17, 0, 3, 2, 18.397418682610855, 10, 0, 4, 0, 0, 0, 0});
      std::cout << "->debe ser 3.0 " << out2[0] << std::endl;
}
