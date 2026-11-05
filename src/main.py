import argparse
from .core.utils import Parser, setup_environment
from .scripts import train, evaluate



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lauch the training / evaluation pipeline within the specified environment.")


    ###
    ### Configure parser argument
    ###
    parser.add_argument("--task", required=True, type=str, choices=["train", "evaluate"], help="Specify the task to perform: 'train' or 'evaluate'.")
    parser.add_argument("--config", required=True, type=str, help="Path to the configuration file. (Supported formats: .yaml, .json)")


    args = parser.parse_args()

    TASK = args.task

    ###
    ### Extract the configuration file path to variables
    ###

    config_file_path = args.config
    config_type = config_file_path.split('.')[-1]

    ## Get the configs
    configs = Parser.parse(config_file_path, config_type)

    ## Environment setup
    environment = setup_environment(configs)


    ## Launch the specified task
    if TASK == "train":
        train(environment)
        
        print("Training completed successfully.")
        
    elif TASK == "evaluate":
        evaluate(environment)
        
        print("Evaluation completed successfully.")

    else:
        raise ValueError(f"Invalid task specified: {TASK}. Please choose either 'train' or 'evaluate'.")
