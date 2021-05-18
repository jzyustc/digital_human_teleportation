import torch
import torch.nn as nn

# matting_loss = nn.BCELoss(size_average=True)


matting_loss = nn.L1Loss()


def muti_matting_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
	loss0 = matting_loss(d0, labels_v)
	loss1 = matting_loss(d1, labels_v)
	loss2 = matting_loss(d2, labels_v)
	loss3 = matting_loss(d3, labels_v)
	loss4 = matting_loss(d4, labels_v)
	loss5 = matting_loss(d5, labels_v)
	loss6 = matting_loss(d6, labels_v)

	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
	'''
	print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (
		loss0.data.item(), loss1.data.item(), loss2.data.item(), loss3.data.item(), loss4.data.item(),
		loss5.data.item(),
		loss6.data.item()))
	'''

	return loss0, loss


def normPRED(d):
	'''
	normalize the predicted mask
	'''
	ma = torch.max(d)
	mi = torch.min(d)
	return (d - mi) / (ma - mi)
