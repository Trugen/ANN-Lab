from utils.pyt_utils import load_model
import jittor as jt

from VAN.jimm.models.van_jittor import van_base, van_large, van_small, van_tiny


def Seg_Model(num_classes, img_size = 512, pretrained=False, pretrained_model=None, recurrence=2, **kwargs):
    assert 'van_size' in kwargs
    assert kwargs['van_size'] in ['van_base', 'van_large', 'van_small', 'van_tiny']
    
    model = eval(kwargs['van_size'])(num_classes=num_classes, img_size=img_size, pretrained=pretrained, recurrence=recurrence)

    # print(model)
    
    if pretrained_model is not None:
        model = load_model(model, pretrained_model)
        # model = jt.load(pretrained_model)

    return model

# if __name__ == '__main__':
    # Seg_Model(19, 512, False, None, van_size='van_base')