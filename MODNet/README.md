# MODNet implement for Human Matting

A fork of [MODNet](https://github.com/ZHKKKe/MODNet) for Human Matting task. 

We refactor the code for our implement, so the raw README.md is deleted.



## Usage

We remove all code to do with training and testing, because we just need to use it instead of training a new model.



### environment

The same as the raw code from MODNet.



### pretrained model

You can get the pretrained model from the raw github page of MODNet, and put them under *./results/models* .



### apply

You can apply your trained model for any image : 

```bash
# before applying, you should adjust the settings items in apply.py
python apply.py
```

- For better performance, we use [Multi-Level foreground estimation](https://arxiv.org/abs/2006.14970) for the foreground image.





