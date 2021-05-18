import os
import json
import time
import torch

# load setting.json
with open("utils/load_settings/test_stage2_settings.json", "r") as file:
	settings = json.load(file)

# load items in setting file
project_name, model_name, pth_name = settings["project_name"], settings["model_name"], settings["pth_name"]
model_path, dataset_path = settings["model_path"], settings["dataset_path"]
batch_size = settings["batch_size"]
image_num, background_num = settings["image_num"], settings["background_num"]
unknown_loss_weight = settings["unknown_loss_weight"]
load_once = settings["dataloader"]["load_once"]
num_worker = settings["dataloader"]["num_worker"]
pin_memory = settings["dataloader"]["pin_memory"]

# set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# create files for logging
with open(model_path + "/test_log.txt", "w") as file:
	content = "-----------------------" + time.strftime("Date: %Y/%m/%d %H:%M:%S",
														time.localtime()) + "-----------------------\n"
	file.write(content)
