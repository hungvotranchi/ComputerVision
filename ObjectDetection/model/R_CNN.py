import torch
import torchvision
from torchvision.models import vgg16
from torchvision.ops import roi_pool

# For image loading and transformations
from torchvision.transforms import functional as F
from PIL import Image

class SimpleRCNN(torch.nn.Module):
    def __init__(self, base_model, num_classes):
        super(SimpleRCNN, self).__init__()
        self.base_model = base_model
        # Assuming ROI size to be 7x7
        self.roi_pool = roi_pool
        self.fc = torch.nn.Linear(512 * 7 * 7, 2048)
        self.classifier = torch.nn.Linear(2048, num_classes)  # num_classes includes background
        self.box_regressor = torch.nn.Linear(2048, num_classes * 4)  # 4 for bbox coordinates

    def forward(self, images, rois, roi_indices):
        """
        images: List of images (as tensors)
        rois: List of RoIs from the images, in (x1, y1, x2, y2) format
        roi_indices: Image index for each ROI
        """
        output = []
        for img in images:
            img_features = self.base_model(img.unsqueeze(0))  # Add batch dimension
            output.append(img_features)
        
        all_features = torch.cat(output, 0)
        pooled_features = self.roi_pool(all_features, rois, output_size=(7, 7))
        
        # Flatten the features
        x = pooled_features.view(pooled_features.size(0), -1)
        x = self.fc(x)
        class_logits = self.classifier(x)
        bbox_regressions = self.box_regressor(x)
        
        return class_logits, bbox_regressions
    

