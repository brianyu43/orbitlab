"""Small global-bottleneck RGB residual predictor and its C4 group average."""
import torch
from torch import nn


class PreviewPredictor(nn.Module):
    def __init__(self,kind):
        super().__init__();assert kind in ['plain','c4'];self.kind=kind
        self.encoder=nn.Sequential(nn.Conv2d(3,16,5,2,2),nn.SiLU(),nn.Conv2d(16,32,5,2,2),nn.SiLU(),
            nn.Conv2d(32,64,5,2,2),nn.SiLU(),nn.Conv2d(64,64,5,2,2),nn.SiLU(),
            nn.Flatten(),nn.Linear(64*8*8,128),nn.SiLU())
        self.expand=nn.Sequential(nn.Linear(128,64*8*8),nn.SiLU())
        self.decoder=nn.Sequential(nn.ConvTranspose2d(64,32,4,2,1),nn.SiLU(),nn.ConvTranspose2d(32,16,4,2,1),nn.SiLU(),
            nn.ConvTranspose2d(16,16,4,2,1),nn.SiLU(),nn.ConvTranspose2d(16,3,4,2,1))
        nn.init.zeros_(self.decoder[-1].weight);nn.init.zeros_(self.decoder[-1].bias)

    def direct(self,image):
        assert image.shape[1:]==(3,128,128)
        z=self.encoder(image);delta=self.decoder(self.expand(z).reshape(-1,64,8,8))
        return (image+torch.tanh(delta)).clamp(0,1)

    def forward(self,image):
        if self.kind=='plain':return self.direct(image)
        n=len(image);orbit=torch.cat([torch.rot90(image,k,(-2,-1)) for k in range(4)],0)
        outputs=self.direct(orbit)
        return torch.stack([torch.rot90(outputs[k*n:(k+1)*n],-k,(-2,-1)) for k in range(4)]).mean(0)
