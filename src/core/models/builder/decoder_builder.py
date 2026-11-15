from .model_builder import ModelBuilder


class EncoderBuilder(ModelBuilder):
    
    
    def build(self, config : dict):
        raise NotImplementedError("The build method must be implemented in subclasses.")