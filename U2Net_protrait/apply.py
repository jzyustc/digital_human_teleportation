from utils import *
from utils.data_loader import *

from model import U2NET

# ------- 0. get settings --------
model_path = "results_stage2/U2Net_L1_e2h/"  # change it
model_name = "u2net"  # change it
pth_name = "u2net_l1_e2h.pth"  # change it

# image
image_name = "comp.png"  # change it
image_path = model_path + "apply/" + image_name  # change it

# set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print('Process image: ', image_path)

# ------- 1. load model --------

net = U2NET(3, 1).to(device)
net.load_state_dict(torch.load(os.path.join(model_path, "model", pth_name)))

# ------- 2. applying process --------

net.eval()

with torch.no_grad():
	image = io.imread(image_path) / 255.

	# image = transform.resize(image, (288, 288), mode='constant')

	image = image / (np.max(image) + 1e-6)

	tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
	if image.shape[2] == 1:
		tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
		tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
		tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
	else:
		tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
		tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
		tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225

	tmpImg = tmpImg.transpose((2, 0, 1))
	inputs = torch.from_numpy(tmpImg).unsqueeze(0).type(torch.FloatTensor).to(device)

	d0, _, _, _, _, _, _ = net(inputs)
	pred = normPRED(d0[0][0])

# ------- 3. output --------

saved_image = ((pred.cpu().data.numpy()) * 255).astype(np.uint8)
im = Image.fromarray(saved_image).convert('RGB')
im.save(model_path + "apply/{}_result.png".format(image_name.split(".")[0]))

print("saved : ", model_path + "apply/{}_result.png".format(image_name.split(".")[0]))
