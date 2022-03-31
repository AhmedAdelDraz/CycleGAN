import torch 
import torchvision
from torchvision import transforms
# from torchsummary import summary

IMAGE_SIZE = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.Resize(int(IMAGE_SIZE*1.33)),
                                 transforms.RandomCrop((IMAGE_SIZE,IMAGE_SIZE)),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5,0.5,0.5),
                                                      (0.5,0.5,0.5))
                                 ])
test_transforms =  transforms.Compose([
                                 transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5,0.5,0.5),
                                                      (0.5,0.5,0.5))
                                 ])

def weights_init_normal(m):
  classname = m.__class__.__name__
  if classname.find("Conv") != -1:
    torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    if hasattr(m, "bias") and m.bias is not None:
      torch.nn.init.constant_(m.bias.data, 0.0)
  elif classname.find("BatchNorm2d") != -1:
    torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
    torch.nn.init.constant_(m.bias.data, 0.0)