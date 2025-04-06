from transformers import AutoModelForCausalLM
from config import MODEL_NAME

class KnowledgeModel(AutoModelForCausalLM):
    """
    Custom model class inheriting from AutoModelForCausalLM.
    This class can be extended to include additional methods or attributes
    specific to the knowledge model.
    """
    def __init__(self, *args, **kwargs):
        self.from_pretrained(MODEL_NAME)
        super().__init__(*args, **kwargs)

    # Additional methods or overrides can be added here if needed.
