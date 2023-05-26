import torch 

class model(torch.nn.Module):
    def __init__(self, in_channels = 3, num_classes = 5):
        super(model, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv1d(
                    in_channels=in_channels, 
                    out_channels = 96, 
                    kernel_size=3, 
                    stride=1, padding=1),
            torch.nn.BatchNorm1d(96),
            torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv1d(96, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU())
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv1d(256, 384, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm1d(384),
            torch.nn.ReLU())
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv1d(384, 384, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm1d(384),
            torch.nn.ReLU())
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv1d(384, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU())
        self.layer6 = torch.nn.Sequential(
            torch.nn.Conv1d(256, 96, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm1d(96),
            torch.nn.ReLU())
        self.layer7 = torch.nn.Sequential(
            torch.nn.Conv1d(96, 31, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm1d(31),
            torch.nn.ReLU())
        
        
    def forward(self, x):
        b, c, w, h = x.shape
        x = x.reshape(b, c, w*h)
        masks = torch.zeros((b, w*h), dtype=torch.bool).to(x.device)

        xm = x.sum(axis=1)
        bi, ind  = torch.where(xm != 0)
        masks[bi, ind] = 1
        
        y = torch.zeros(b, 31, w*h).to(x.device)
        # if masks.sum() !=0:
        out = x[bi, :, ind][..., None]

        if out.shape[0] == 1:
            out = out.repeat(2, 1, 1)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.layer5(out)
            out = self.layer6(out)
            out = self.layer7(out)
            out = out.mean(dim=0, keepdim=True)
            print(out.shape)
        else:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.layer5(out)
            out = self.layer6(out)
            out = self.layer7(out)

        y[bi, :, ind] = out.squeeze(-1)


        return y.reshape(b, 31, w, h)