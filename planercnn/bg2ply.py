from utils_3d.get3d import *


def get3d(args):
	evaluate(args)


if __name__ == '__main__':
	args = parse_args()

	args.keyname = args.dataset
	args.test_dir = 'results/3d_models/'

	# python bg2ply.py --methods=f --suffix=warping_refine --dataset=inference --customDataFolder=plancercnn/example_images
	get3d(args)
