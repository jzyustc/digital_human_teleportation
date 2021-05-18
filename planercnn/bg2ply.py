from utils.get3d import *


args = parse_args()

args.keyname = args.dataset
args.test_dir = 'results/3d_models/'

# python bg2ply.py --methods=f --suffix=warping_refine --dataset=inference --customDataFolder=plancercnn/example_images
evaluate(args)
