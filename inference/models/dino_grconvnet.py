import torch
import torch.nn as nn
import torch.nn.functional as F

from inference.models.grasp_model import GraspModel, ResidualBlock
from groundingdino.util.slconfig import SLConfig

class DinoNormalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], device: str = "cuda"):
        super(DinoNormalize, self).__init__()
        if len(mean) != 3 or len(std) != 3:
            raise ValueError("Mean and std must each have three elements for RGB channels.")
        self.mean = torch.tensor(mean).view(3, 1, 1).to(device)
        self.std = torch.tensor(std).view(3, 1, 1).to(device)

    def forward(self, x):
        if x.dim() != 4 or x.size(1) != 3:
            raise ValueError("Input tensor must be in the shape (B, 3, H, W)")
        x = x.float()  # Ensure the input is float for proper division
        return (x - self.mean) / self.std

from inference.models.groundingdino.modeling.groundingdino import build_groundingdino
from inference.models.groundingdino.util.misc import clean_state_dict

def create_dino_bone(model_config_path: str, model_checkpoint_path: str, device: str = "cuda"):
    # Get the config
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    # Create the model
    dino_bone = build_groundingdino(args)
    # Get the model pretrained path
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    # Load the pretrained data
    state_dict = clean_state_dict(checkpoint["model"])
    state_dict.pop('label_enc.weight', None)
    state_dict.pop('bert.embeddings.position_ids', None)
    dino_bone.load_state_dict(state_dict, strict=True)
    dino_bone = dino_bone.to(device)
    # Freeze all layers in the model
    dino_bone.eval()
    for param in dino_bone.parameters():
        param.requires_grad = False
    return dino_bone

class DINO_GenerativeConvNet(GraspModel):

    def __init__(self, input_channels=4, output_channels=1, channel_size=32, dropout=False, prob=0.0, is_aligned=0):
        super(DINO_GenerativeConvNet, self).__init__()
        # self.conv1 = nn.Conv2d(input_channels, channel_size, kernel_size=9, stride=1, padding=4)
        # self.bn1 = nn.BatchNorm2d(channel_size)

        # self.conv2 = nn.Conv2d(channel_size, channel_size * 2, kernel_size=4, stride=2, padding=1)
        # self.bn2 = nn.BatchNorm2d(channel_size * 2)

        # self.conv3 = nn.Conv2d(channel_size * 2, channel_size * 4, kernel_size=4, stride=2, padding=1)
        # self.bn3 = nn.BatchNorm2d(channel_size * 4)

        # self.res1 = ResidualBlock(channel_size * 4, channel_size * 4)
        # self.res2 = ResidualBlock(channel_size * 4, channel_size * 4)
        # self.res3 = ResidualBlock(channel_size * 4, channel_size * 4)
        # self.res4 = ResidualBlock(channel_size * 4, channel_size * 4)
        # self.res5 = ResidualBlock(channel_size * 4, channel_size * 4)

        self.preprocess_image = DinoNormalize()

        # the frozen, pretrained dino bone
        self.dino_bone = create_dino_bone("./inference/models/groundingdino/config/GroundingDINO_SwinB_cfg.py", 
                                          "./inference/models/groundingdino/groundingdino_swinb_cogcoor.pth")
        # the post-processor for dino to use the model cut
        self.postprocess_dino_conv = nn.ConvTranspose2d(1024, channel_size * 16, kernel_size=2, stride=2)
        self.postprocess_dino_bn = nn.BatchNorm2d(channel_size * 16)

        # cutoff transpose
        self.conv4 = nn.ConvTranspose2d(channel_size * 16, channel_size * 8, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(channel_size * 8)

        self.conv5 = nn.ConvTranspose2d(channel_size * 8, channel_size, kernel_size=4, stride=2, padding=2)
        self.bn5 = nn.BatchNorm2d(channel_size)

        self.conv6 = nn.ConvTranspose2d(channel_size, channel_size, kernel_size=9, stride=1, padding=4)

        if is_aligned:
            self.pos_output = nn.Conv2d(in_channels=channel_size, out_channels=256, kernel_size=2)
        else:
            self.pos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.cos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.sin_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.width_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)

        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

        self.is_aligned = is_aligned
        assert(self.is_aligned != 0)


    def forward(self, x_in, prompts):
        x = self.preprocess_image(x_in)
        x, prompt_feat = self.dino_bone(x, captions=prompts)
        x = F.relu(self.postprocess_dino_bn(self.postprocess_dino_conv(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)
        x = F.interpolate(x, size=(225, 225), mode='bilinear')
        
        if self.dropout:
            pos_output = self.pos_output(self.dropout_pos(x))
            cos_output = self.cos_output(self.dropout_cos(x))
            sin_output = self.sin_output(self.dropout_sin(x))
            width_output = self.width_output(self.dropout_wid(x))
        else:
            pos_output = self.pos_output(x)
            cos_output = self.cos_output(x)
            sin_output = self.sin_output(x)
            width_output = self.width_output(x)

        try:
            if self.is_aligned == 1:
                # normalize pos_output at channel level
                pos_output = F.normalize(pos_output, p=2, dim=1)
                # normalize prompt_feat
                prompt_feat = torch.mean(prompt_feat, dim=1)[:,:,None,None]
                prompt_feat = F.normalize(prompt_feat, p=2, dim=1)
                # perform alignment for new pos
                pos_output = torch.sum(pos_output*prompt_feat, dim=1, keepdims=True)
        except:
            pass
        
        return pos_output, cos_output, sin_output, width_output
