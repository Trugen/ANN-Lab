# unchecked
import os
import argparse

import jittor as jt

from utils.logger import get_logger
from utils.pyt_utils import extant_file


logger = get_logger()


class Engine(object):
    def __init__(self, custom_parser=None):
        logger.info(
            "Jittor Version {}".format(jt.__version__))
        self.devices = None
        self.distributed = False

        if custom_parser is None:
            self.parser = argparse.ArgumentParser()
        else:
            assert isinstance(custom_parser, argparse.ArgumentParser)
            self.parser = custom_parser

        self.inject_default_parser()
        self.args = self.parser.parse_args()

        self.continue_state_object = self.args.continue_fpath

        gpus = os.environ["CUDA_VISIBLE_DEVICES"]
        # self.devices =  [i for i in range(len(gpus.split(',')))] 
        self.devices = [int(i) for i in gpus.split(',')]
        print(self.devices)

    def inject_default_parser(self):
        p = self.parser
        p.add_argument('-d', '--devices', default='',
                       help='set data parallel training')
        p.add_argument('-c', '--continue', type=extant_file,
                       metavar="FILE",
                       dest="continue_fpath",
                       help='continue from one certain checkpoint')
        p.add_argument('--local_rank', default=0, type=int,
                       help='process rank on node')

    def data_parallel(self, model):
        return model

    def get_train_loader(self, train_dataset):
        return train_dataset, None

    def get_test_loader(self, test_dataset):
        return test_dataset, None


    def all_reduce_tensor(self, tensor, norm=True):
        return jt.mean(tensor)


    def __enter__(self):
        return self


    def __exit__(self, type, value, tb):
        if type is not None:
            logger.warning(
                "A exception occurred during Engine initialization, "
                "give up running process")
            return False
