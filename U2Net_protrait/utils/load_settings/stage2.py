import os
import json
import time
import torch

# load setting.json
with open("utils/load_settings/train_stage2_settings.json", "r") as file:
	settings = json.load(file)

# load items in setting file
project_name, model_name, pth_name = settings["project_name"], settings["model_name"], settings["pth_name"]
model_path, dataset_path = settings["model_path"], settings["dataset_path"]
batch_size, epoch_num = settings["batch_size"], settings["epoch_num"]
image_num, background_num, easy_background_num = settings["image_num"], settings["background_num"], settings[
	"easy_background_num"]
unknown_loss_weight = settings["unknown_loss_weight"]
update_ratio_epoch = settings["update_ratio_epoch"]
load_once = settings["dataloader"]["load_once"]
num_worker = settings["dataloader"]["num_worker"]
pin_memory = settings["dataloader"]["pin_memory"]

# set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# create files for logging

full_project_name = time.strftime(project_name + "_{}__%Y_%m_%d__%H_%M_%S".format(unknown_loss_weight),
								  time.localtime()) + "/"
result_dir = os.path.join("results_stage2", full_project_name)
if not os.path.exists(result_dir): os.mkdir(result_dir)

model_dir = os.path.join(result_dir, "model")
if not os.path.exists(model_dir): os.mkdir(model_dir)

image_dir = os.path.join(result_dir, "image")
if not os.path.exists(image_dir): os.mkdir(image_dir)

with open("utils/load_settings/train_stage2_settings.json", "r") as file:
	settings_json = file.read()
	file.close()

with open(result_dir + "/train_settings.json", "w") as file:
	file.write(settings_json)
	file.close()

with open(result_dir + "/train_log.txt", "w") as file:
	content = "-----------------------" + time.strftime("Date: %Y/%m/%d %H:%M:%S",
														time.localtime()) + "-----------------------\n"
	file.write(content)
	file.close()

with open(result_dir + "/val_log.txt", "w") as file:
	content = "-----------------------" + time.strftime("Date: %Y/%m/%d %H:%M:%S",
														time.localtime()) + "-----------------------\n"
	file.write(content)
	file.close()
