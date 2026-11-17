from .base_builder import BaseBuilder


class DecoderBuilder(BaseBuilder):
    
    
    def build(self, config : dict):
        raise NotImplementedError("The build method must be implemented in subclasses.")