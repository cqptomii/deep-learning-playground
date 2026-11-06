from core.optimizer_registry import OptimizerRegistry
from core.scheduler_registry import SchedulerRegistry
from core.tasks import get_task
from core.data import get_dataset
from core.models import get_model
from core.visualization import process_visualizers
from core.engine import Trainer
import os

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
    COMPUTE_VISUALIZATIONS = environment.get("compute_visualizations")
    USE_GRAD_CLIP = environment.get("use_grad_clip", False)
    ## Create the optimizer
    optimizer = environment.get("optimizer")
    optimizer = OptimizerRegistry.get_optimizer(optimizer)
    
    ## Create the scheduler if it's provided
    scheduler = environment.get("scheduler")
    if scheduler:
        scheduler = SchedulerRegistry.get_scheduler(scheduler, optimizer)
    else:
        scheduler = None
    
    ## Setup the output_dir
    OUTPUT_DIR = environment.get("output_dir")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
    os.makedirs(LOG_DIR, exist_ok=True)
    VISU_DIR = os.path.join(OUTPUT_DIR, "visualizations")
    os.makedirs(VISU_DIR, exist_ok=True)
    
    
    task = environment.get("task")
    train_metrics = environment.get("train_metrics")
    test_metrics = environment.get("validation_metrics")
    loss_fn = environment.get("loss_function")
    
    ### Create the task based on the configuration
    task = get_task(task, train_metrics, test_metrics, loss_fn)
    
    ## Create the dataset based on the configuration
    dataset = environment.get("dataset")
    train_dataloader, validation_dataloader, test_dataloader = get_dataset(dataset)
    
    
    ## Create the model based on the configuration
    model = environment.get("model")
    model = get_model(model)
    
    ## Create visualizers based on the configuration
    visualization = environment.get("visualization")
    visualization = process_visualizers(visualization, task)
    
    ## Create callbacks 
    callbacks = environment.get("callbacks")
    callbacks = task.get_callbacks(callbacks, visualizers=visualization)
    
    
    ## Create the trainer
    trainer = Trainer(
        model=model,
        task=task,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        test_dataloader=test_dataloader,
        callbacks=callbacks,
        device=DEVICE,
        grad_clip=USE_GRAD_CLIP,
    )
    
    ## Train the model
    history = trainer.fit(epochs=EPOCHS)
    
    ## Test the model on the test set
    test_results = trainer.test()
    
    
    
    
    
    