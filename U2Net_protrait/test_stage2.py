from torch.utils.data import DataLoader
from torchvision import transforms

from utils import *
from utils.data_loader import *

from model import U2NET

# ------- 0. get settings --------
from utils.load_settings.test_stage2 import *

# ------- 1. training dataset --------

tra_img_name_list, tra_lbl_name_list, tra_bg_name_list, tra_ez_bg_name_list = paixin_comp_mod(dataset_path)

train_num = int(0.9 * min(len(tra_img_name_list), image_num))
all_num = int(min(len(tra_img_name_list), image_num))

bg_train_num = int(0.9 * min(len(tra_bg_name_list), background_num))
bg_all_num = int(min(len(tra_bg_name_list), background_num))

print("---")
print("test results: ", all_num - train_num)
print("test backgrounds: ", bg_all_num - bg_train_num)
print("---")

val_dataset = PaixinDataset(
	img_name_list=tra_img_name_list[train_num:all_num],
	lbl_name_list=tra_lbl_name_list[train_num:all_num],
	bg_name_list=tra_bg_name_list[bg_train_num:bg_all_num],
	ez_bg_name_list=tra_ez_bg_name_list,
	transform=transforms.Compose([RandomCrop(288), ToTensor()]),
	load_once=load_once)
val_dataset.epoch = 1000
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker,
							pin_memory=pin_memory)

# ------- 2. define model and optimizer --------

net = U2NET(3, 1).to(device)
net.load_state_dict(torch.load(os.path.join(model_path, "model", pth_name)))

# ------- 4. training process --------
print("---start training...")

'''
validation
'''
net.eval()

saved_id = np.random.randint(0, len(val_dataloader) - 1)
ite_num = 0
running_loss = 0.0
running_unknown_loss = 0.0
running_tar_loss = 0.0
for i, data in enumerate(val_dataloader):
	ite_num = ite_num + 1

	with torch.no_grad():

		inputs, labels, trimap = data['image'], data['label'], data["trimap"].to(device)

		inputs = inputs.type(torch.FloatTensor).to(device)
		labels = labels.type(torch.FloatTensor).to(device)

		d0, d1, d2, d3, d4, d5, d6 = net(inputs)
		loss2, loss = muti_matting_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels)

		loss_unknown = matting_loss(d0 * trimap, labels * trimap)
		loss += unknown_loss_weight * loss_unknown

		# saved id
		if i == saved_id:
			pred = normPRED(d0.repeat(1, 3, 1, 1))
			tmpImg = torch.zeros_like(inputs)
			tmpImg[:, 0] = (inputs[:, 0] * 0.229 + 0.485)
			tmpImg[:, 1] = (inputs[:, 1] * 0.224 + 0.456)
			tmpImg[:, 2] = (inputs[:, 2] * 0.224 + 0.406)
			inputs_masked = tmpImg * pred
			saved = torch.cat([tmpImg, labels.repeat(1, 3, 1, 1), pred, inputs_masked], 2)
			b, c, h, w = saved.shape
			saved = saved.permute(2, 0, 3, 1).reshape(h, b * w, c)

		# statistics
		running_loss += loss.data.item()
		running_tar_loss += loss2.data.item()
		running_unknown_loss += loss_unknown.data.item()

		with open(model_path + "/test_log.txt", "a") as file:
			content = "image %d : %3f, unknown loss: %3f, tar: %3f \n" % \
					  (i, loss.data.item(), loss_unknown.data.item(), loss2.data.item())
			file.write(content)
			print(content)

		# del temporary outputs and loss
		del d0, d1, d2, d3, d4, d5, d6, loss2, loss

'''
validation result
'''
with open(model_path + "/test_log.txt", "a") as file:
	content = "test loss: %3f, unknown loss: %3f, tar: %3f \n" % (
		running_loss / ite_num, running_unknown_loss / ite_num, running_tar_loss / ite_num)
	file.write(content)
	print(content)

saved_image = ((saved.cpu().data.numpy()) * 255).astype(np.uint8)
im = Image.fromarray(saved_image).convert('RGB')
im.save(model_path + "image/test.png")
