# PlaneRCNN implement for 3D reconstruction

A fork of [PlaneRCNN](https://github.com/NVlabs/planercnn) to detects and reconstructs piece-wise planar surfaces from a single RGB image.

We add some codes to expand the implement, so the raw README.md is deleted.





## Our Adding

We add a program to apply the PlaneRCNN and output as a .ply file.

- **3D model generating** : Based on the raw evaluate function, we reconstruct the 3D model from a single RGB image and resolute the raw output API, then save the reformatted XYZ  and RGBA data in a **.ply** file.



We also add a program to render the .ply file and inpainting the gap part.

- **3D model rendering** : We implement a **Render** class to resolute .ply files generated in former step, and render it by **vertexes** or **faces** to a single RGB image, a gap mask and a depthmap. 
  - The faces-based rendering is much slower than by vertexes, while it's much finer when the camera is close to the model.
  - We also provide a function to render by open3d. It's much faster, but because of the insufficiency of API, we can't obtain depthmap, which makes it hard to guarantee the perspective effect in image compositing.

- **Image Inpainting** : Because of the lack of information of the reconstructed 3D model, many gaps exists in the rendered RGB image and depth map (marked by the gap mask). We provide two inpainting methods implemented on numpy to inpainting them.





## Usage



### environment

The same as the raw code from PlaneRCNN. Pytorch <= 0.4.0 is needed for compiling of Mask-RCNN. 

During my running, I find that Pytorch >= 0.4.1 makes the code unavailable, even though in the raw Github page of PlaneRCNN, the author says it works.



### pretrained model

You can get the pretrained model from the raw Github page of PlaneRCNN, and put them under *./results/models* .



### apply : rgb image to ply file

You can apply your trained model for any image to generate a .ply file: 

```bash
# before applying, you should adjust the settings items in apply.py
python bg2ply.py --methods=f --suffix=warping_refine --dataset=inference --customDataFolder=plancercnn/example_images
```



### apply :  ply file to background

You can render a background image from the .ply file : 

```bash
# before applying, you should adjust the settings items in apply.py
python ply2bg.py 
```





