from torchvision.transforms import v2
from typing import Callable

class TransformRegistry:
    """
    A registry for retrieving a transform function based on its name. This class maintains a mapping of transform names to their corresponding callable functions.
    """
    
    @staticmethod
    def parse_transform(
        transform_name: str, 
        parameters: dict | None = None
    ) -> Callable:
        """
        Retrieve a transform function based on its name.

        Args:
            transform_name (str): The name of the transform to retrieve.

        Returns:
            callable: The corresponding transform function.

        Raises:
            ValueError: If the transform name is not found in the registry.
        """
        
        ## Get the Callable fonction
        transform_class = getattr(v2, transform_name, None)
        
        ## Handle the case where the transform is not found in the registry
        if transform_class is None:
            raise ValueError(
                f"Unknown torchvision transform: {transform_name}"
            )
        
        ## Check if the retrieved transform is a valid torchvision v2 transform
        if not isinstance(transform_class, type) or not issubclass(
            transform_class, v2.Transform
        ):
            raise ValueError(
                f"{transform_name} is not a valid torchvision v2 transform"
            )
        
        return transform_class(**(parameters or {}))
    
    @staticmethod
    def get_transforms(transform_list : list) -> list:
        """Method which retrieve a transform compose based on a list of transform

        Args:
            transform_list (list): A list of transform names to be retrieved and composed.

        Returns:
            list: A list of transform functions that applies the specified transforms in sequence.
        """
        transforms = []
        for transform_name in transform_list:
            transform = TransformRegistry.parse_transform(transform_name)
            transforms.append(transform())
        
        return transforms
        