# Awesome-Image-Text-to-3D [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

A curated list of papers and open-source resources focused on text/image to 3D.

## Table of Contents
- [Dataset](#dataset)

- [Survey](#survey)

- [Unified framework for 3D content generation](#unified-framework-for-3D-content-generation)

- [Single Image to 3D](#single-image-to-3d)

- [Text to 3D](#text-to-3d)


## Dataset
## 2023
### Objaverse-XL: A Universe of 10M+ 3D Objects

**Author**: Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, Eli VanderBilt, Aniruddha Kembhavi, Carl Vondrick, Georgia Gkioxari, Kiana Ehsani, Ludwig Schmidt, Ali Farhadi

**Date**: 11 Jul 2023

<details span>
<summary><b>Abstract</b></summary>
Natural language processing and 2D vision models have attained remarkable proficiency on many tasks primarily by escalating the scale of training data. However, 3D vision tasks have not seen the same progress, in part due to the challenges of acquiring high-quality 3D data. In this work, we present Objaverse-XL, a dataset of over 10 million 3D objects. Our dataset comprises deduplicated 3D objects from a diverse set of sources, including manually designed objects, photogrammetry scans of landmarks and everyday items, and professional scans of historic and antique artifacts. Representing the largest scale and diversity in the realm of 3D datasets, Objaverse-XL enables significant new possibilities for 3D vision. Our experiments demonstrate the improvements enabled with the scale provided by Objaverse-XL. We show that by training Zero123 on novel view synthesis, utilizing over 100 million multi-view rendered images, we achieve strong zero-shot generalization abilities. We hope that releasing Objaverse-XL will enable further innovations in the field of 3D vision at scale.
</details>

[📄 Paper](https://arxiv.org/pdf/2307.05663)| [Project](https://objaverse.allenai.org/) | [💻 Code](https://github.com/allenai/objaverse-xl) | [🤗 Demo](https://colab.research.google.com/drive/15XpZMjrHXuky0IgBbXcsUtb_0g-XWYmN?usp=sharing)

## 2022
### Objaverse: A Universe of Annotated 3D Objects

**Author**: Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, Ali Farhadi

**Date**: 15 Dec 2022

<details span>
<summary><b>Abstract</b></summary>
Massive data corpora like WebText, Wikipedia, Conceptual Captions, WebImageText, and LAION have propelled recent dramatic progress in AI. Large neural models trained on such datasets produce impressive results and top many of today's benchmarks. A notable omission within this family of large-scale datasets is 3D data. Despite considerable interest and potential applications in 3D vision, datasets of high-fidelity 3D models continue to be mid-sized with limited diversity of object categories. Addressing this gap, we present Objaverse 1.0, a large dataset of objects with 800K+ (and growing) 3D models with descriptive captions, tags, and animations. Objaverse improves upon present day 3D repositories in terms of scale, number of categories, and in the visual diversity of instances within a category. We demonstrate the large potential of Objaverse via four diverse applications: training generative 3D models, improving tail category segmentation on the LVIS benchmark, training open-vocabulary object-navigation models for Embodied AI, and creating a new benchmark for robustness analysis of vision models. Objaverse can open new directions for research and enable new applications across the field of AI.
</details>

[📄 Paper](https://arxiv.org/pdf/2307.05663)| [Project](https://objaverse.allenai.org/objaverse-1.0/)


## Survey
## 2024
### A Survey On Text-to-3D Contents Generation In The Wild

**Author**: Chenhan Jiang

**Date**: 15 May 2024

<details span>
<summary><b>Abstract</b></summary>
3D content creation plays a vital role in various applications, such as gaming, robotics simulation, and virtual reality. However, the process is labor-intensive and time-consuming, requiring skilled designers to invest considerable effort in creating a single 3D asset. To address this challenge, text-to-3D generation technologies have emerged as a promising solution for automating 3D creation. Leveraging the success of large vision language models, these techniques aim to generate 3D content based on textual descriptions. Despite recent advancements in this area, existing solutions still face significant limitations in terms of generation quality and efficiency. In this survey, we conduct an in-depth investigation of the latest text-to-3D creation methods. We provide a comprehensive background on text-to-3D creation, including discussions on datasets employed in training and evaluation metrics used to assess the quality of generated 3D models. Then, we delve into the various 3D representations that serve as the foundation for the 3D generation process. Furthermore, we present a thorough comparison of the rapidly growing literature on generative pipelines, categorizing them into feedforward generators, optimization-based generation, and view reconstruction approaches. By examining the strengths and weaknesses of these methods, we aim to shed light on their respective capabilities and limitations. Lastly, we point out several promising avenues for future research. With this survey, we hope to inspire researchers further to explore the potential of open-vocabulary text-conditioned 3D content creation.
</details>

[📄 Paper](https://arxiv.org/pdf/2305.06131)

## 2023
### Generative AI meets 3D: A Survey on Text-to-3D in AIGC Era

**Author**: Chenghao Li, Chaoning Zhang, Atish Waghwase, Lik-Hang Lee, Francois Rameau, Yang Yang, Sung-Ho Bae, Choong Seon Hong

**Date**: 10 Jun 2024

<details span>
<summary><b>Abstract</b></summary>
Generative AI (AIGC, a.k.a. AI generated content) has made significant progress in recent years, with text-guided content generation being the most practical as it facilitates interaction between human instructions and AIGC. Due to advancements in text-to-image and 3D modeling technologies (like NeRF), text-to-3D has emerged as a nascent yet highly active research field. Our work conducts the first comprehensive survey and follows up on subsequent research progress in the overall field, aiming to help readers interested in this direction quickly catch up with its rapid development. First, we introduce 3D data representations, including both Euclidean and non-Euclidean data. Building on this foundation, we introduce various foundational technologies and summarize how recent work combines these foundational technologies to achieve satisfactory text-to-3D results. Additionally, we present mainstream baselines and research directions in recent text-to-3D technology, including fidelity, efficiency, consistency, controllability, diversity, and applicability. Furthermore, we summarize the usage of text-to-3D technology in various applications, including avatar generation, texture generation, shape editing, and scene generation.
</details>

[📄 Paper](https://arxiv.org/pdf/2305.06131)


## Unified framework for 3D content generation

- [threestudio](https://github.com/threestudio-project/threestudio)

<p align="center">
    <picture>
    <img alt="threestudio" src="https://user-images.githubusercontent.com/19284678/236847132-219999d0-4ffa-4240-a262-c2c025d15d9e.png" width="50%">
    </picture>
</p>

## Single Image to 3D
## 2024
### TripoSR: Fast 3D Object Reconstruction from a Single Image

**Author**: Dmitry Tochilkin, David Pankratz, Zexiang Liu, Zixuan Huang, Adam Letts, Yangguang Li, Ding Liang, Christian Laforte, Varun Jampani, Yan-Pei Cao

**Date**: 4 Mar 2024

<details span>
<summary><b>Abstract</b></summary>
This technical report introduces TripoSR, a 3D reconstruction model leveraging transformer architecture for fast feed-forward 3D generation, producing 3D mesh from a single image in under 0.5 seconds. Building upon the LRM network architecture, TripoSR integrates substantial improvements in data processing, model design, and training techniques. Evaluations on public datasets show that TripoSR exhibits superior performance, both quantitatively and qualitatively, compared to other open-source alternatives. Released under the MIT license, TripoSR is intended to empower researchers, developers, and creatives with the latest advancements in 3D generative AI.
</details>

[📄 Paper](https://arxiv.org/pdf/2403.02151) | [💻 Code](https://github.com/VAST-AI-Research/TripoSR) | [🤗 Demo](https://huggingface.co/spaces/stabilityai/TripoSR) | [🎬 Model](https://huggingface.co/stabilityai/TripoSR)


### AGG: Amortized Generative 3D Gaussians for Single Image to 3D

**Authors**: Dejia Xu, Ye Yuan, Morteza Mardani, Sifei Liu, Jiaming Song, Zhangyang Wang, Arash Vahdat

**Date**: 8 Jan 2024

<details span>
<summary><b>Abstract</b></summary>
Given the growing need for automatic 3D content creation pipelines, various 3D representations have been studied to generate 3D objects from a single image. Due to its superior rendering efficiency, 3D Gaussian splatting-based models have recently excelled in both 3D reconstruction and generation. 3D Gaussian splatting approaches for image to 3D generation are often optimization-based, requiring many computationally expensive score-distillation steps. To overcome these challenges, we introduce an Amortized Generative 3D Gaussian framework (AGG) that instantly produces 3D Gaussians from a single image, eliminating the need for per-instance optimization. Utilizing an intermediate hybrid representation, AGG decomposes the generation of 3D Gaussian locations and other appearance attributes for joint optimization. Moreover, we propose a cascaded pipeline that first generates a coarse representation of the 3D data and later upsamples it with a 3D Gaussian super-resolution module. Our method is evaluated against existing optimization-based 3D Gaussian frameworks and sampling-based pipelines utilizing other 3D representations, where AGG showcases competitive generation abilities both qualitatively and quantitatively while being several orders of magnitude faster.
</details>

[📄 Paper](https://arxiv.org/pdf/2401.04099) | [🌐 Project Page](https://ir1d.github.io/AGG/) | [💻 Code (not yet)]()


## 2023
### [ICLR '2024] LRM: Large Reconstruction Model for Single Image to 3D

**Author**: Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, Hao Tan

**Date**: 8 Nov 2023

<details span>
<summary><b>Abstract</b></summary>
We propose the first Large Reconstruction Model (LRM) that predicts the 3D model of an object from a single input image within just 5 seconds. In contrast to many previous methods that are trained on small-scale datasets such as ShapeNet in a category-specific fashion, LRM adopts a highly scalable transformer-based architecture with 500 million learnable parameters to directly predict a neural radiance field (NeRF) from the input image. We train our model in an end-to-end manner on massive multi-view data containing around 1 million objects, including both synthetic renderings from Objaverse and real captures from MVImgNet. This combination of a high-capacity model and large-scale training data empowers our model to be highly generalizable and produce high-quality 3D reconstructions from various testing inputs, including real-world in-the-wild captures and images created by generative models. Video demos and interactable 3D meshes can be found on our LRM project webpage: this https URL.
</details>

[📄 Paper](https://arxiv.org/pdf/2311.04400) | [🌐 Project Page](https://yiconghong.me/LRM/) | [💻 Code (not yet)]()


### Zero123++: a Single Image to Consistent Multi-view Diffusion Base Model

**Author**: Ruoxi Shi, Hansheng Chen, Zhuoyang Zhang, Minghua Liu, Chao Xu, Xinyue Wei, Linghao Chen, Chong Zeng, Hao Su

**Date**: 23 Oct 2023

<details span>
<summary><b>Abstract</b></summary>
We report Zero123++, an image-conditioned diffusion model for generating 3D-consistent multi-view images from a single input view. To take full advantage of pretrained 2D generative priors, we develop various conditioning and training schemes to minimize the effort of finetuning from off-the-shelf image diffusion models such as Stable Diffusion. Zero123++ excels in producing high-quality, consistent multi-view images from a single image, overcoming common issues like texture degradation and geometric misalignment. Furthermore, we showcase the feasibility of training a ControlNet on Zero123++ for enhanced control over the generation process.
</details>

[📄 Paper](https://arxiv.org/pdf/2310.15110) | [🤗 Demo](https://huggingface.co/spaces/sudo-ai/zero123plus-demo-space) | [💻 Code](https://github.com/SUDO-AI-3D/zero123plus)


### [ICCV '2023]Zero-1-to-3: Zero-shot One Image to 3D Object

**Author**: Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, Carl Vondrick

**Date**: 20 Mar 2023 

<details span>
<summary><b>Abstract</b></summary>
We introduce Zero-1-to-3, a framework for changing the camera viewpoint of an object given just a single RGB image. To perform novel view synthesis in this under-constrained setting, we capitalize on the geometric priors that large-scale diffusion models learn about natural images. Our conditional diffusion model uses a synthetic dataset to learn controls of the relative camera viewpoint, which allow new images to be generated of the same object under a specified camera transformation. Even though it is trained on a synthetic dataset, our model retains a strong zero-shot generalization ability to out-of-distribution datasets as well as in-the-wild images, including impressionist paintings. Our viewpoint-conditioned diffusion approach can further be used for the task of 3D reconstruction from a single image. Qualitative and quantitative experiments show that our method significantly outperforms state-of-the-art single-view 3D reconstruction and novel view synthesis models by leveraging Internet-scale pre-training.
</details>

[📄 Paper](https://arxiv.org/pdf/2303.11328) | [🌐 Project Page](https://zero123.cs.columbia.edu/) | [💻 Code](https://github.com/cvlab-columbia/zero123?tab=readme-ov-file)


## Text to 3D
## 2022
### DreamFusion: Text-to-3D using 2D Diffusion

**Author**: Ben Poole, Ajay Jain, Jonathan T. Barron, Ben Mildenhall

**Date**: 29 Sep 2022

<details span>
<summary><b>Abstract</b></summary>
Recent breakthroughs in text-to-image synthesis have been driven by diffusion models trained on billions of image-text pairs. Adapting this approach to 3D synthesis would require large-scale datasets of labeled 3D data and efficient architectures for denoising 3D data, neither of which currently exist. In this work, we circumvent these limitations by using a pretrained 2D text-to-image diffusion model to perform text-to-3D synthesis. We introduce a loss based on probability density distillation that enables the use of a 2D diffusion model as a prior for optimization of a parametric image generator. Using this loss in a DeepDream-like procedure, we optimize a randomly-initialized 3D model (a Neural Radiance Field, or NeRF) via gradient descent such that its 2D renderings from random angles achieve a low loss. The resulting 3D model of the given text can be viewed from any angle, relit by arbitrary illumination, or composited into any 3D environment. Our approach requires no 3D training data and no modifications to the image diffusion model, demonstrating the effectiveness of pretrained image diffusion models as priors.
</details>

[📄 Paper](https://arxiv.org/pdf/2209.14988) | [🌐 Project Page](https://dreamfusion3d.github.io/)
