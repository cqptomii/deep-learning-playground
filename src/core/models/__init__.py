from torch import nn

from .builder import build
from .model_zoo import ModelZoo
from .model_registry import MODEL_REGISTRY


def get_model(model_config : dict) -> nn.Module:
    
    ## Extract the model name
    name = model_config.get("name")

    ## Extract additional parameters for the model
    loading_type = model_config.get("type", "torchvision")
    use_pretrained = model_config.get("use_pretrained", False)
    weights_param = model_config.get("weights", {"name": ""})
    
    
    ## Load the model base on the configuration
    if loading_type == "torchvision":
        weight_name = weights_param.get("name", "")
        model = ModelZoo.get_model(
            model_name=name, 
            use_pretrained=use_pretrained,
            weights=weight_name
        )
        num_classes = model_config.get("num_classes")
        if num_classes is not None and hasattr(model, "fc"):
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    elif loading_type == "custom":
        
        ## Check if there is a configuration file given
        if "config_path" in model_config :
            config_path : str = model_config["config_path"]
            format = config_path.split(".")[-1]
            
            model_config = build(
                path=config_path, 
                format=format
            )
            
        else:
            ## Load the model_fn from the registry
            if name not in MODEL_REGISTRY:
                available = ", ".join(MODEL_REGISTRY.keys())
                raise ValueError(
                    f"Unknown model '{name}'. Available models: {available}"
                )
            else:
                model_fn, model_parameters = MODEL_REGISTRY[name]
                
                if use_pretrained:
                    weight_path = weights_param.get("weight_path", None)
                    if weight_path is None:
                        raise ValueError("Weight path must be provided for custom models when 'use_pretrained' is True.")
                else:
                    weight_path = None
                
                model = model_fn(
                    use_pretrained=use_pretrained, 
                    weight_path=weight_path,
                    **model_parameters
                )    
        
        
    else:
        raise ValueError(f"Unknown loading type '{loading_type}'. Supported types: 'torchvision', 'custom'.")

    return model


__version__ = "1.0.0"
__all__ = [
    "MODEL_REGISTRY",
    "ModelZoo",
    "get_model",
]