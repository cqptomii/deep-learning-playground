from networkx import config
import yaml, json

class Parser:
    
    ###
    ### Load methods
    ###
    
    @staticmethod
    def _parse_yaml(path : str):
        """
        Parse a YAML configuration file.

        Args:
            data (str): The path to the YAML configuration file.

        Returns:
            dict: The parsed configuration as a dictionary.
        """
        with open(path, 'r') as file:
            return yaml.safe_load(file)
    
    @staticmethod
    def _parse_json(path : str):
        """
        Parse a JSON configuration file.

        Args:
            data (str): The path to the JSON configuration file.

        Returns:
            dict: The parsed configuration as a dictionary.
        """
        with open(path, 'r') as file:
            return json.load(file)
    
    ###
    ### Parse method
    ###
    
    @staticmethod
    def _setup_environment(self, config : dict):
        """Method to ensure that the configuration file contain all parameters needed
    
        Args:
            config (dict): Extracted configuration from the YAML file. 
        """
            
        ##
        ## CHECK REQUIRED PARAMETERS PRESENCE
        ##
            
        ## Check if the 'model' section is present
        if "model" not in config:
            raise ValueError("Missing 'model' section.")
        if not isinstance(config["model"], dict):
            raise ValueError("'model' must be a dictionary.")
            
        ## Ensure that the 'model' section contains the required parameters
        model = config["model"]
        if "model_name" not in model:
            raise ValueError("Missing 'model_name' in 'model' section.")
        if "use_pretrained" not in model:
            model["use_pretrained"] = False  ## Default value
        else:
            if model["use_pretrained"] not in [True, False]:
                raise ValueError("'use_pretrained' must be a boolean value.")
            
            ## Check that the weights section is present
            if "weights" not in model:
                raise ValueError("Missing 'weights' in 'model' section.")
            if not isinstance(model["weights"], dict):
                raise ValueError("'weights' must be a dictionary with this structure: {name: str, weight_path: str}" )
            
        if "type" not in model:
            raise ValueError("Missing 'type' in 'model' section.")
            
            
        ## Check if the 'dataset' section is present
        if "dataset" not in config:
            raise ValueError("Missing 'dataset' section.")
        if not isinstance(config["dataset"], dict):
            raise ValueError("'dataset' must be a dictionary.")
        
        ## Ensure that the 'dataset' section contains the required parameters
        dataset = config["dataset"]
        if "name" not in dataset:
            raise ValueError("Missing 'name' in 'dataset' section.")
        if "data_path" not in dataset:
            raise ValueError("Missing 'data_path' in 'dataset' section.")
         
        ## Check if the 'task' section is present
        if "task" not in config:
            raise ValueError("Missing 'task' section.")
        if not isinstance(config["task"], dict):
            raise ValueError("'task' must be a dictionary.")
        
        ## Ensure that the 'task' section contains the required parameters
        task = config["task"]
        if "name" not in task:
            raise ValueError("Missing 'name' in 'task' section.")
        if "loss_function" not in task:
            raise ValueError("Missing 'loss_function' in 'task' section.")
        if "train_metrics" not in task:
            raise ValueError("Missing 'train_metrics' in 'task' section.")
        if "validation_metrics" not in task:
            raise ValueError("Missing 'validation_metrics' in 'task' section.")    
           
            
        ##
        ## SETUP ENVIRONMENT variables with default values if not present in the config
        ##
        config["batch_size"] = config.get("batch_size", 32)
        config["epochs"] = config.get("epochs", 20)
        config["device"] = config.get("device", "cpu") ## By default use CPU        
        config["optimizer"] = config.get("optimizer", {"name": "adam", "lr": 0.001})
        config["output_dir"] = config.get("output_dir", "./outputs/")
        config["compute_visualizations"] = config.get("compute_visualizations", False)
        config["callbacks"] = config.get("callbacks", {"early_stopping": {"monitor": "val_loss"}})
        config["scheduler"] = config.get("scheduler", {})
            
        return config


    @staticmethod
    def parse(path : str, config_type : str):
        """
        Parse the configuration file based on the specified type.

        Args:
            path (str): The path to the configuration file.
            config_type (str): The type of the configuration file (e.g., 'yaml', 'json').

        Returns:
            dict: The parsed configuration as a dictionary.
        """
        if config_type == "yaml":
            return Parser._parse_yaml(path)
            
        elif config_type == "json":
            return Parser._parse_json(path)
        else:
            raise ValueError(f"Unsupported configuration type: {config_type}")
        