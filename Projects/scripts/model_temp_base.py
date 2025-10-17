# model_temp.py || model_temp_base.py

from transformers import  AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
import torch.nn as nn

def get_model_and_tokenizer(config):
    """Tải pre-trained model và tokenizer."""
    print(f"Đang tải model: {config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    # Tải config/model/tokenizer với trust_remote_code=True để cho phép model custom
    cfg = AutoConfig.from_pretrained(config.MODEL_NAME, trust_remote_code=True)
    print(f"Model config: {cfg}")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME, 
        num_labels=len(config.LABEL_MAP)
    )

    # apply classifier dropout if provided in config
    if hasattr(config, 'CLASSIFIER_DROPOUT'):
        if hasattr(model.config, 'classifier_dropout'):
            model.config.classifier_dropout = config.CLASSIFIER_DROPOUT
        if hasattr(model.config, 'hidden_dropout_prob'):
            model.config.hidden_dropout_prob = config.CLASSIFIER_DROPOUT
            
        if hasattr(model.config, 'attention_probs_dropout_prob'):
            model.config.attention_probs_dropout_prob = min(0.15, max(0.1, config.CLASSIFIER_DROPOUT))
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = config.CLASSIFIER_DROPOUT
    return model, tokenizer

