from core.optimizer_registry import OptimizerRegistry
from core.scheduler_registry import SchedulerRegistry
from core.tasks import get_task
from core.data import get_dataset
from core.data.vision import build_split_transforms
from core.models import get_model
from core import get_callback
from core.visualization import process_visualizers
from core.engine import Trainer
import os
import torch

def train(environment : dict):
    """
    Train the model based on the provided environment.

    Args:
        environment (dict): The environment containing configurations and settings for training.
    """
    
    ### Extract Hyperparameters and configurations from the environment
    
    BATCH_SIZE = environment.get("batch_size")
    EPOCHS = environment.get("epochs")
    DEVICE = environment.get("device")

    ## Check if the device is available
    if DEVICE == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Training on CPU.")
        DEVICE = "cpu"
    
    USE_GRAD_CLIP = environment.get("use_grad_clip", False)
    
    ## Setup the output_dir
    OUTPUT_DIR = environment.get("output_dir")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
    os.makedirs(LOG_DIR, exist_ok=True)
    VISU_DIR = os.path.join(OUTPUT_DIR, "visualizations")
    os.makedirs(VISU_DIR, exist_ok=True)
    
    
    ##
    ## Task Creation
    ##
    
    task = environment.get("task")
    
    task_type = task.get("type", "classification")
    train_metrics = task.get("metrics", {}).get("train", ["accuracy"])
    test_metrics = task.get("metrics", {}).get("test", ["accuracy", "confusion_matrix"])
    loss_fn = task.get("loss_fn")
    
    ### Create the task based on the configuration
    task = get_task(task_type, train_metrics, test_metrics, loss_fn)
    
    ##
    ## Dataset Creation
    ##
    
    ## Create the dataset based on the configuration
    augmentation = environment.get("data_augmentation", {})
    train_transform, eval_transform = build_split_transforms(
        augmentation.get("train"),
        augmentation.get("validation"),
    )
    dataset_config = {
        **environment["dataset"],
        "batch_size": BATCH_SIZE,
        "num_workers": environment.get("num_workers", 0),
        "train_transform": train_transform,
        "eval_transform": eval_transform,
    }
    
    ## Get the train, validation, and test dataloaders
    train_dataloader, validation_dataloader, test_dataloader = get_dataset(dataset_config)
    
    
    ##
    ## Model Creation
    ##
    
    ## Create the model based on the configuration
    model_config : dict = environment.get("model", {})
    model = get_model(model_config)
    
    ##
    ## Optimizer, Scheduler and Callbacks Creation
    ##
    
    ## Create the optimizer
    optimizer_config = environment.get("optimizer")
    optimizer_config = optimizer_config.copy()
    optimizier_name = optimizer_config.get("name")
    optimizer_config.pop("name", None)
    
    optimizer = OptimizerRegistry.get_optimizer(name=optimizier_name, model_parameters=model.parameters(), **optimizer_config)
    
    ## Create the scheduler if it's provided
    scheduler_config = environment.get("scheduler", {})
    scheduler_name = scheduler_config.get("name", None)
    scheduler_config = scheduler_config.copy()
    scheduler_config.pop("name", None)
    if scheduler_name:
        scheduler = SchedulerRegistry.get_scheduler(name=scheduler_name, optimizer=optimizer, **scheduler_config)
    else:
        scheduler = None
        
    
    ## Create visualizers based on the configuration
    visualization = environment.get("visualizers")
    visualization = process_visualizers(visualization)
    
    ## Create callbacks 
    callbacks = environment.get("callbacks")
    callbacks = task.get_callbacks(callbacks, visualizers=visualization, save_path=VISU_DIR)
    
    
    ## Create the trainer
    trainer = Trainer(
        model=model,
        task=task,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_dataloader,
        val_loader=validation_dataloader,
        test_loader=test_dataloader,
        callbacks=callbacks,
        device=DEVICE,
        grad_clip=USE_GRAD_CLIP,
        epochs=EPOCHS
    )
    
    ## Train the model
    history = trainer.fit()
    
    ## Test the model on the test set
    test_results = trainer.test()
    
    
    
    
    
    