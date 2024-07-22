
from .net.pvt  import Net
from pytorch_grad_cam.utils.image import show_cam_on_image
import uuid
import torch
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from PIL import Image
import torch.nn as nn
imageSize=512
# 创建一个转换，将图像转换为 tensor
transform = transforms.Compose([
    transforms.Resize([imageSize, imageSize]),
    transforms.ToTensor()

])

class HotMap(nn.Module):
    def __init__(self):
        super(HotMap, self).__init__()
        self.net=Net()
        self.fc = nn.Linear(imageSize*imageSize, 2)
        self.net.load_state_dict(torch.load('D:\pythonpoject\DTNet\\bestEpoch\\55-0.0211-1pvt_epoch_best.pth'))

    def forward(self, x):
        _,_,x,_=self.net(x)

        x = torch.flatten(x, 1)
        x=self.fc(x)
        return x



def vis_cam():
    model = HotMap()
    #
    model.eval()
    src = 'D:\pythonpoject\DTNet\Dataset\TestDataset\CAMO\Imgs\camourflage_00061.jpg'
    model.cuda()
    img = Image.open(src)
    print(f'The Image size:{img.size}')

    img_tensor = transform(img)

    print(f'The shape of image preprocessed: {img_tensor.shape}')

    grad_cam = GradCAM(model=model, target_layers=[model.net.gcam1,model.net.gcam3])
    imgt= img_tensor.unsqueeze(0)
    cam = grad_cam(input_tensor=imgt)  # 输入的Shape: B x C x H x W

    print(f'Cam.shape: {cam.shape}')
    print(f'Cam.max: {cam.max()}, Cam.min: {cam.min()}')
    input_tensor = img_tensor.permute(1, 2, 0).numpy()
    vis = show_cam_on_image(input_tensor, cam[0], use_rgb=False)
    print(img.size)

    vis_img = Image.fromarray(vis)

    vis_img2=vis_img.resize(img.size)
    vis_img2.save(f'cam_{uuid.uuid1()}.jpg')
    return vis


