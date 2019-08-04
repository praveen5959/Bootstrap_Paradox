#------------------------------------------------------------------------------
#	Libraries
#------------------------------------------------------------------------------
import torch, argparse
from models import BiSeNet,UNet,ICNet
from models.DeepLab import DeepLabV3Plus
from base import VideoInference

#------------------------------------------------------------------------------
#   Argument parsing
#------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Arguments for the script")

parser.add_argument('--use_cuda', action='store_true', default=True,
                    help='Use GPU acceleration')

parser.add_argument('--input_size', type=int, default=320,
                    help='Input size')

parser.add_argument('--checkpoint', type=str, default="model_best.pth",
                    help='Path to the trained model file')

args = parser.parse_args()


#------------------------------------------------------------------------------
#	Main execution
#------------------------------------------------------------------------------
# Build model
#model = UNet(num_classes=2)
#model =BiSeNet(backbone='resnet18',num_classes=2)
#model =ICNet(backbone='resnet18',num_classes=2)
model =DeepLabV3Plus(backbone='resnet18',num_classes=2)
trained_dict = torch.load(args.checkpoint, map_location="cpu")['state_dict']
model.load_state_dict(trained_dict, strict=False)
model.eval()


# Inference
inference = VideoInference(
    model=model,
    video_path=0,
    input_size=args.input_size,
    draw_mode='matting',
)
inference.run()
