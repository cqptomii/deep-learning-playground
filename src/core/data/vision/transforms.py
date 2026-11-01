from torchvision.transforms import v2

class TransformsWrapper:
    """
    A wrapper class for handling training and validation transforms.
    """
    def __init__(self, train_transforms : list = [], val_transforms : list = []):
        """ Initialization of the Wrapper class
        By default, transforms are empty lists

        Args:
            train_transforms (list): Transforms list for training data
            val_transforms (list): Transforms list for validation data
        """
        self.train_transforms : list  = train_transforms
        self.val_transforms : list = val_transforms
    
    
    def __add_to_transforms(self, transform, transforms_list : list, index : int = None):
        """Method to add a specific transform into a specific list of transforms (train or validation).
        By default, the transform is added to the end of the list.
        If an index is provided, the transform is inserted at that index.
        
        Args:
            transform (_type_): The transform to be added to the list.
            transforms_list (list): The list of transforms to which the new transform will be added.
            index (int, optional): The index at which to insert the transform. Defaults to None.
        """
        if index is not None:
            ## Check if the index is valid
            if index < 0 or index > len(transforms_list):
                print(f"Warning: Index {index} is out of bounds for train_transforms list. Appending to the end instead.")
                        
                transforms_list.append(transform)
            else:
                transforms_list.insert(index, transform)
        else:
            transforms_list.append(transform)
    
    
    def add_train_transform(self, transform, index : int = None):
        """Method to add a specific transform into the train transform list

        Args:
            transform (_type_): The transform to be added to the train transforms list.
            index (int, optional): The index at which to insert the transform. Defaults to None.
        """
        self.__add_to_transforms(transform, self.train_transforms, index)
    def add_val_transform(self, transform, index : int = None):
        """Method to add a specific transform into the validation transform list
        
        Args:
            transform (_type_): The transform to be added to the validation transforms list.
            index (int, optional): The index at which to insert the transform. Defaults to None.
        """
        self.__add_to_transforms(transform, self.val_transforms, index)
    
    ###
    ### SETTER Functions
    ###
    
    def set_train_transforms(self, transforms : list):
        """Method to set the train transforms

        Args:
            transforms (list): List of transforms to be set for training data
        """
        self.train_transforms = transforms
    def set_val_transforms(self, transforms : list):
        """Method to set the validation transforms
        
        Args:
            transforms (list): List of transforms to be set for validation data
        """
        self.val_transforms = transforms
    
    
    ###
    ### GETTER Functions
    ###
    
    def get_train_transforms(self):
        return v2.Compose(
            self.train_transforms
        ) 
    def get_val_transforms(self):
        return v2.Compose(
            self.val_transforms
        )
    
         
        
    