import os 
import argparse

file_dir = os.path.dirname(__file__)

class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Evaluation options")

        self.parser.add_argument('--save_dir', type=str, default=os.path.join(file_dir, 'SavedModels', 'Run2'), help='directory to save or load from')
        self.parser.add_argument('--model', type=str, default='model.pth', help='model name')
        self.parser.add_argument('--load_state_dict', action='store_true', help='set true to load model as a state dict')
        self.parser.add_argument('--use_vgg_old', action='store_true', help='use hand-written VGG architecture')

        self.parser.add_argument('--finetune_model_name', type=str, help='name to give finetuned model')
	
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
