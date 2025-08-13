from ultralytics import YOLO
import torch
from torch.quantization import quantize_dynamic

class QuantizedYOLO:
    def __init__(self, model_path, quantize_mode='fp16'):
        self.original_model = YOLO(model_path)
        self.quantize_mode = quantize_mode
        self.model = self._prepare_quantized_model()
        
    def _prepare_quantized_model(self):
        model = self.original_model.model
        model.eval()
        
        if self.quantize_mode == 'fp16':
            model = model.half()
        elif self.quantize_mode == 'int8':
            model = quantize_dynamic(
                model,
                {torch.nn.Linear, torch.nn.Conv2d},
                dtype=torch.qint8
            )
        return model
        
    def __call__(self, *args, **kwargs):
        # Convert input to right dtype
        if self.quantize_mode == 'fp16':
            kwargs['imgsz'] = kwargs.get('imgsz', 640)
            kwargs['half'] = True
        return self.original_model(*args, **kwargs)
