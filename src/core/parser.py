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
            
        ## Check if the 'dataset' section is present
        if "dataset" not in config:
            raise ValueError("Missing 'dataset' section.")
        if not isinstance(config["dataset"], dict):
            raise ValueError("'dataset' must be a dictionary.")
            
        ## Check if the 'task' section is present
        if "task" not in config:
            raise ValueError("Missing 'task' section.")
        if not isinstance(config["task"], dict):
            raise ValueError("'task' must be a dictionary.")
            
            
        ##
        ## SETUP ENVIRONMENT variables with default values if not present in the config
        ##
        BATCH_SIZE = config.get("batch_size", 32)
        EPOCHS = config.get("epochs", 20)
        DEVICE = config.get("device", "cpu") ## By default use CPU        
        OPTIMIZER = config.get("optimizer", {"name": "adam", "lr": 0.001})
        LOGGING = config.get("logging_dir", "path/to/logs")
        COMPUTE_VISUALIZATIONS = config.get("compute_visualizations", False)
        CALLBACKS = config.get("callbacks", {"early_stopping": {"monitor": "val_loss"}})
        
            
        return {
            "model" : config["model"],
            "dataset" : config["dataset"],
            "task" : config["task"],
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "device": DEVICE,
            "optimizer": OPTIMIZER,
            "logging_dir": LOGGING,
            "compute_visualizations": COMPUTE_VISUALIZATIONS,
            "CALLBACKS": CALLBACKS
        }
        
        
        
        
        
        
        pass


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
        