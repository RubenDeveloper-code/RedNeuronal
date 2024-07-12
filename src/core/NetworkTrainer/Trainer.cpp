#include "../../../include/core/NetworkTrainer/Trainer.hpp"
#include "../../../include/alerts/messages.hpp"
#include "../../../include/core/NetworkTrainer/Checks.hpp"
#include "../../../include/core/NetworkTrainer/Iteration.hpp"
#include "../../../include/core/NetworkTrainer/ModelCheckpoint.hpp"
#include "../../../include/core/NetworkTrainer/ModelLoss.hpp"
#include "../../../include/core/NetworkTrainer/TrainerUI.hpp"
#include <chrono>
#include <memory>

Trainer::Trainer(Network &network, SharedResources &shared_resources,
                 TrainSpects &train_spects, AlgorithmsSpects &algorithms_spects,
                 std::shared_ptr<LossFuctions::LossFunction> loss_funcion)
    : network(network), shared_resources(shared_resources),
      trainer_spects(train_spects), loss_funcion(loss_funcion),
      model_loss(loss_funcion),
      iteration(network,
                std::move(SetterData(train_spects.dataset.training_dataset)),
                model_loss, train_spects.mini_batch),
      model_ckpt(network, train_spects.checkpoints_spects),
      checks(network, train_spects.earlystop_spects),
      trainer_ui(train_spects.epochs) {
      initAlgorithms(algorithms_spects);
}

Trainer::Status Trainer::fit() {
      status = Status::FITTING;
      double loss, validation_loss;
      Messages::Message({"training model..."});
      int &epoch_it = *shared_resources.epochs_it;
      while ((epoch_it)++ < trainer_spects.epochs) {
            auto start = std::chrono::high_resolution_clock::now();
            if (alpha_algorithm != nullptr)
                  alpha_algorithm->run();
            loss = iteration.iterate();
            validation_loss = iteration.validationForwardPropagation(
                trainer_spects.dataset.validation_dataset);
            trainer_ui.updateBar(epoch_it, loss, validation_loss);
            model_ckpt.adminCheckpoint(epoch_it, validation_loss);
            if (checks.reachUmbrall(loss, trainer_spects.umbral, epoch_it))
                  status = Status::DONE;
            if (checks.reachPatience(validation_loss))
                  status = Status::RELOAD;
            auto end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(end - start).count();
            trainer_ui.displayElapsedTime(elapsed *
                                          (trainer_spects.epochs - epoch_it));
            if (status != Status::FITTING) {
                  trainer_ui.endBar();
                  return status;
            }
      }
      trainer_ui.endBar();
      Messages::Message({"trained model"});
      return Status::DONE;
}

void Trainer::initAlgorithms(AlgorithmsSpects &algorithms_spects) {
      alpha_algorithm = AlphaAlgorithms::newInstance(
          static_cast<AlphaAlgorithms::TYPE>(algorithms_spects.alphaModifier),
          algorithms_spects.args_alpha_modifier);
}
