from planercnn.utils.get3d import *

'''
STEP 2.0 : Reconstruct a 3D model from a single rgb image
'''

if __name__ == "__main__":
	args = parse_args()

	args.keyname = args.dataset
	results_path = "results/"  # change it
	args.test_dir = results_path + '3d_models/'

	# python pre_bg2ply.py --methods=f --suffix=warping_refine --dataset=inference --customDataFolder=plancercnn/example_images
	evaluate(args)
