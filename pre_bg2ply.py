from planercnn.bg2ply import get3d

'''
STEP 2.0 : Reconstruct a 3D model from a single rgb image
'''

if __name__ == "__main__":
	args = parse_args()

	results_path = "results/"  # change it

	args.methods = 'f'
	args.suffix = 'warping_refine'
	args.dataset = 'inference'
	args.customDataFolder = 'plancercnn/example_images'
	args.keyname = args.dataset
	args.test_dir = results_path + '3d_models/'

	get3d(args)
