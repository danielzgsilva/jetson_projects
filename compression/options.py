import os 
import argparse

file_dir = os.path.dirname(__file__)

class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Evaluation options")

        self.parser.add_argument('--model', type=str, help='model to evaluate')
	self.parser.add_argument('--finetune_model_name', type=str)
	
        self.parser.add_argument('--n', type=int, default=5, help='Number of times to evaluate on test set. Results are averaged over all runs')
        
        self.parser.add_argument('--tensorRT', action='store_true', help='use TensorRT to optimize model')
        
        self.parser.add_argument('--compress', action='store_true', help='use basis filter compression algorithm')
        self.parser.add_argument('--use_weights', type=bool, default=True)
        self.parser.add_argument('--add_bn', type=bool, default=True)
        self.parser.add_argument('--fixed_basbs', type=bool, default=True)
        self.parser.add_argument('--compress_factor', type=float, default=0.8)

    def parse(self):
        self.opts = self.parser.parse_args()

        return self.opts
