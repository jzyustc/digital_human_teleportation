# iDIH implement for Image Harmonization

A fork of [iDIH](https://github.com/saic-vul/image_harmonization) for Image Harmonization task.

We add some codes to expand the implement, so the raw README.md is deleted.





## Our Adding

**A harmonization API for harmonization between a RGB foreground and a RGBD background** : Based on the perspective relationship, we allow the foreground to be placed on anywhere in the background, and then apply the harmonization. 





## Usage

We remove all code to do with training and testing, because we just need to use it instead of training a new model.



### environment

The same as the raw code from iDIH.



### pretrained model

You can get the pretrained model from the raw GitHub page of iDIH, and put them under *./results/models* .



### apply

You can apply your trained model for any image : 

```bash
# before applying, you should adjust the settings items in apply.py
python apply.py hrnet18s_idih256 results/models/hrnet18s_idih256.pth
```



