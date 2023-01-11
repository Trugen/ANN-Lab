# done
import os
import jittor as jt
from jittor import Module, nn
from PIL import Image
import numpy as np

def INF(B,H,W):
    return -jt.diag(jt.array(float("inf")).repeat(H),0).unsqueeze(0).repeat(B*W,1,1)


class CrissCrossAttention(Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim):
        super(CrissCrossAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax()
        self.INF = INF
        self.gamma = jt.zeros(1)    # nn.Parameter(jt.zeros(1))


    def execute(self, x):
        # ! Removed .contiguous()
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).view(m_batchsize*height,-1,width)
        energy_H = (jt.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = jt.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(jt.concat([energy_H, energy_W], 3))      # [B, H, W, H+W]

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).view(m_batchsize*width,height,height)
        #print(concate)
        #print(att_H) 
        att_W = concate[:,:,:,height:height+width].view(m_batchsize*height,width,width)
        out_H = jt.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = jt.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        
        # th = 10
        # tw = 10
        # attention = np.zeros((height, width))
        # concate_np = concate.numpy()
        # concate_np[0, th, tw, -width:]
        # print(concate_np[0, th, tw, -width:])
        # for w in range(width):
        #     attention[th, :] += concate_np[0, th, w, -width:]
        #     attention[:, w] += concate_np[0, th, w, :height]
        # for h in range(height):
        #     attention[:, tw] += concate_np[0, h, tw, :height]
        #     attention[h, :] += concate_np[0, h, tw, -width:]
        # img = Image.fromarray(np.uint8(attention * 20))
        # img.save("attention.png")
        
        return self.gamma*(out_H + out_W) + x


class HorizontalVerticalAttention(Module):
    # TODO: only checked the correctness of input-output dimension
    """ Horizontal-Vertical Attention Module"""
    def __init__(self, in_dim):
        super(HorizontalVerticalAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax()
        self.INF = INF
        self.gamma_H = jt.zeros(1)
        self.gamma_W = jt.zeros(1)

    def execute(self, x):
        residual = x
        m_batchsize, _, height, width = x.shape     # [B, C_in, H, W]
        
        proj_query_H = self.query_conv(x).permute(0,3,1,2).view(m_batchsize*width,-1,height).permute(0, 2, 1)       # [B*W, H, C_out]
        proj_key_H = self.key_conv(x).permute(0,3,1,2).view(m_batchsize*width,-1,height)                            # [B*W, C_out, H]
        proj_value_H = self.value_conv(x).permute(0,3,1,2).view(m_batchsize*width,-1,height)                        # [B*W, C_in, H]
        energy_H = (jt.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).\
                    view(m_batchsize,width,height,height)                                                           # [B, W, H, H]
        att_H = self.softmax(energy_H).view(m_batchsize*width,height,height)                                        # [B*W, H, H]
        out_H = jt.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)     # [B, C_in, H, W]
        
        x = self.gamma_H*out_H + residual
        residual = x
        proj_query_W = self.query_conv(x).permute(0,2,1,3).view(m_batchsize*height,-1,width).permute(0, 2, 1)       # [B*H, W, C_out]
        proj_key_W = self.key_conv(x).permute(0,2,1,3).view(m_batchsize*height,-1,width)                            # [B*H, C_out, W]
        proj_value_W = self.value_conv(x).permute(0,2,1,3).view(m_batchsize*height,-1,width)                        # [B*H, C_in, W]
        energy_W = (jt.bmm(proj_query_W, proj_key_W) + self.INF(m_batchsize, width, height)).\
                    view(m_batchsize,height,width,width)                                                            # [B, H, W, W]
        att_W = self.softmax(energy_W).view(m_batchsize*height,width,width)                                         # [B*H, W, W]
        out_W = jt.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)     # [B, C_in, H, W]
        
        return self.gamma_W*out_W + residual      # [B, C_in, H, W]


if __name__ == '__main__':
    model = CrissCrossAttention(64)
    x = jt.randn(2, 64, 5, 6)
    out = model(x)
    print(out.shape)
