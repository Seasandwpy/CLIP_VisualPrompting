# CLIP_VisualPrompting

### Unofficial Implementation of the paper [What does CLIP know about a red circle? Visual prompt engineering for VLMs](https://openaccess.thecvf.com/content/ICCV2023/papers/Shtedritski_What_does_CLIP_know_about_a_red_circle_Visual_prompt_ICCV_2023_paper.pdf)

### prerequisites: Installation  following [CLIP](https://github.com/openai/CLIP) repo.

### Usage

We have the example of the pan_1.png and pan_2.png, and match them to texts ["an image of the handle of a pan",
"an image of the cooking area of a pan"]. After running the script, we have a probability of [[0.6423, 0.3527],
        [0.3517, 0.6433]] as the final scores.

###  Acknowledgement
We borrow the optimal transport function from [SuperGlue](https://arxiv.org/abs/1911.11763) 
