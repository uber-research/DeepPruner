# DeepPruner: Learning Efficient Stereo Matching via Differentiable PatchMatch

This repository releases code for our paper [DeepPruner: Learning Efficient Stereo Matching via Differentiable PatchMatch](https://arxiv.org/abs/1909.05845).

##### Table of Contents  
[DeepPruner](#DeepPruner)  
[Differentiable Patch Match](#DifferentiablePatchMatch)  
[Requirements (Major Dependencies)](#Requirements)  
[Citation](#Citation)  


  <a name="DeepPruner"></a>
  ###  **DeepPruner** 

 + An efficient "Real Time Stereo Matching" algorithm, which takes as input 2 images and outputs a disparity (or depth) map.
	
	
	
	![](readme_images/DeepPruner.png)
	
	
	
 + Results/ Metrics:
							
    + [**KITTI**](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo): **Results competitive to SOTA, while being real-time (8x faster than SOTA). SOTA among published real-time algorithms**.
	
    
    ![](readme_images/KITTI_test_set.png)
	  ![](readme_images/CRP.png)
    ![](readme_images/uncertainty_vis.png)
	
	
	  + [**ETH3D**](https://www.eth3d.net/low_res_two_view?mask=all&metric=bad-2-0): **SOTA among all ROB entries**. 
	  
    + **SceneFlow**: **2nd among all published algorithms, while being 8x faster than the 1st.**
	  
    <p align="center">
    <img src="readme_images/sceneflow.png" width="60%" />
	  </p> 
         
    
	
	  + [**Robust Vision Challenge**](http://www.robustvision.net/index.php): **Overall ranking 1st**. 
	   
    <p align="center">
    <img src="readme_images/rob.png" width="60%" />
	  </p>
	
	+ Runtime: **62ms** (for DeepPruner-fast), **180ms** (for DeepPruner-best)
	  
	+ Cuda Memory Requirements: **805MB** (for DeepPruner-best)



  <a name="DifferentiablePatchMatch"></a>
  ### **Differentiable Patch Match**
  + Fast algorithm for finding dense nearest neighbor correspondences between patches of images regions. 
    Differentiable version of the generalized Patch Match algorithm. ([Barnes et al.](https://gfx.cs.princeton.edu/pubs/Barnes_2010_TGP/index.php))
    
   <p>
   <img src="readme_images/DPM.png" width="50%" /> <img src="readme_images/DPM_filters.png" width="40%" />
    </p>

More details in the corresponding folder README.


<a name="Requirements"></a>
## Requirements (Major Dependencies)
+ Pytorch (0.4.0+)
+ Python2.7
+ torchvision (0.2.0+)



<a name="Citation"></a>
## Citation

If you use our source code, or our paper, please consider citing the following:
> @inproceedings{Duggal2019ICCV,  
title = {DeepPruner: Learning Efficient Stereo Matching  via Differentiable PatchMatch},  
author = {Shivam Duggal and Shenlong Wang and Wei-Chiu Ma and Rui Hu and Raquel Urtasun},  
booktitle = {ICCV},  
year = {2019}
}

Correspondences to Shivam Duggal <shivamduggal.9507@gmail.com>, Shenlong Wang <slwang@cs.toronto.edu>, Wei-Chiu Ma <weichium@mit.edu>
