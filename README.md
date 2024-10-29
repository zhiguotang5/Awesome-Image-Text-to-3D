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

[📄 Paper](https://arxiv.org/pdf/2307.05663)| [🌐 Project Page](https://objaverse.allenai.org/) | [💻 Code](https://github.com/allenai/objaverse-xl) | [🤗 Demo](https://colab.research.google.com/drive/15XpZMjrHXuky0IgBbXcsUtb_0g-XWYmN?usp=sharing)

## 2022
### Objaverse: A Universe of Annotated 3D Objects

**Author**: Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, Ali Farhadi

**Date**: 15 Dec 2022

<details span>
<summary><b>Abstract</b></summary>
Massive data corpora like WebText, Wikipedia, Conceptual Captions, WebImageText, and LAION have propelled recent dramatic progress in AI. Large neural models trained on such datasets produce impressive results and top many of today's benchmarks. A notable omission within this family of large-scale datasets is 3D data. Despite considerable interest and potential applications in 3D vision, datasets of high-fidelity 3D models continue to be mid-sized with limited diversity of object categories. Addressing this gap, we present Objaverse 1.0, a large dataset of objects with 800K+ (and growing) 3D models with descriptive captions, tags, and animations. Objaverse improves upon present day 3D repositories in terms of scale, number of categories, and in the visual diversity of instances within a category. We demonstrate the large potential of Objaverse via four diverse applications: training generative 3D models, improving tail category segmentation on the LVIS benchmark, training open-vocabulary object-navigation models for Embodied AI, and creating a new benchmark for robustness analysis of vision models. Objaverse can open new directions for research and enable new applications across the field of AI.
</details>

[📄 Paper](https://arxiv.org/pdf/2307.05663)| [🌐 Project Page](https://objaverse.allenai.org/objaverse-1.0/)


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
### DreamCraft3D++: Efficient Hierarchical 3D Generation with Multi-Plane Reconstruction Model

**Author**: Jingxiang Sun, Cheng Peng, Ruizhi Shao, Yuan-Chen Guo, Xiaochen Zhao, Yangguang Li, Yanpei Cao, Bo Zhang, Yebin Liu

**Date**: 16 Oct 2024 

<details span>
<summary><b>Abstract</b></summary>
We introduce DreamCraft3D++, an extension of DreamCraft3D that enables efficient high-quality generation of complex 3D assets. DreamCraft3D++ inherits the multi-stage generation process of DreamCraft3D, but replaces the time-consuming geometry sculpting optimization with a feed-forward multi-plane based reconstruction model, speeding up the process by 1000x. For texture refinement, we propose a training-free IP-Adapter module that is conditioned on the enhanced multi-view images to enhance texture and geometry consistency, providing a 4x faster alternative to DreamCraft3D's DreamBooth fine-tuning. Experiments on diverse datasets demonstrate DreamCraft3D++'s ability to generate creative 3D assets with intricate geometry and realistic 360° textures, outperforming state-of-the-art image-to-3D methods in quality and speed. The full implementation will be open-sourced to enable new possibilities in 3D content creation.
</details>

[📄 Paper](https://arxiv.org/pdf/2410.12928) | [🌐 Project Page](https://dreamcraft3dplus.github.io/) | [💻 Code](https://github.com/MrTornado24/DreamCraft3D_Plus)


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


### [ICLR '2024] DreamCraft3D: Hierarchical 3D Generation with Bootstrapped Diffusion Prior

**Author**: Jingxiang Sun, Bo Zhang, Ruizhi Shao, Lizhen Wang, Wen Liu, Zhenda Xie, Yebin Liu

**Date**: 25 Oct 2023

<details span>
<summary><b>Abstract</b></summary>
We present DreamCraft3D, a hierarchical 3D content generation method that produces high-fidelity and coherent 3D objects. We tackle the problem by leveraging a 2D reference image to guide the stages of geometry sculpting and texture boosting. A central focus of this work is to address the consistency issue that existing works encounter. To sculpt geometries that render coherently, we perform score distillation sampling via a view-dependent diffusion model. This 3D prior, alongside several training strategies, prioritizes the geometry consistency but compromises the texture fidelity. We further propose Bootstrapped Score Distillation to specifically boost the texture. We train a personalized diffusion model, Dreambooth, on the augmented renderings of the scene, imbuing it with 3D knowledge of the scene being optimized. The score distillation from this 3D-aware diffusion prior provides view-consistent guidance for the scene. Notably, through an alternating optimization of the diffusion prior and 3D scene representation, we achieve mutually reinforcing improvements: the optimized 3D scene aids in training the scene-specific diffusion model, which offers increasingly view-consistent guidance for 3D optimization. The optimization is thus bootstrapped and leads to substantial texture boosting. With tailored 3D priors throughout the hierarchical generation, DreamCraft3D generates coherent 3D objects with photorealistic renderings, advancing the state-of-the-art in 3D content generation. 
</details>

[📄 Paper](https://arxiv.org/pdf/2310.16818) | [🌐 Project Page](https://mrtornado24.github.io/DreamCraft3D/) | [💻 Code](https://github.com/deepseek-ai/DreamCraft3D)


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
## 2023

### [CVPR '2024 Highlight] RichDreamer: A Generalizable Normal-Depth Diffusion Model for Detail Richness in Text-to-3D

**Author**: Lingteng Qiu, Guanying Chen, Xiaodong Gu, Qi Zuo, Mutian Xu, Yushuang Wu, Weihao Yuan, Zilong Dong, Liefeng Bo, Xiaoguang Han

**Date**: 28 Nov 2023 

<details span>
<summary><b>Abstract</b></summary>
Lifting 2D diffusion for 3D generation is a challenging problem due to the lack of geometric prior and the complex entanglement of materials and lighting in natural images. Existing methods have shown promise by first creating the geometry through score-distillation sampling (SDS) applied to rendered surface normals, followed by appearance modeling. However, relying on a 2D RGB diffusion model to optimize surface normals is suboptimal due to the distribution discrepancy between natural images and normals maps, leading to instability in optimization. In this paper, recognizing that the normal and depth information effectively describe scene geometry and be automatically estimated from images, we propose to learn a generalizable Normal-Depth diffusion model for 3D generation. We achieve this by training on the large-scale LAION dataset together with the generalizable image-to-depth and normal prior models. In an attempt to alleviate the mixed illumination effects in the generated materials, we introduce an albedo diffusion model to impose data-driven constraints on the albedo component. Our experiments show that when integrated into existing text-to-3D pipelines, our models significantly enhance the detail richness, achieving state-of-the-art results.
</details>



### Instant3D: Fast Text-to-3D with Sparse-View Generation and Large Reconstruction Model

**Author**: Jiahao Li, Hao Tan, Kai Zhang, Zexiang Xu, Fujun Luan, Yinghao Xu, Yicong Hong, Kalyan Sunkavalli, Greg Shakhnarovich, Sai Bi

**Date**: 10 Nov 2023

<details span>
<summary><b>Abstract</b></summary>
Text-to-3D with diffusion models has achieved remarkable progress in recent years. However, existing methods either rely on score distillation-based optimization which suffer from slow inference, low diversity and Janus problems, or are feed-forward methods that generate low-quality results due to the scarcity of 3D training data. In this paper, we propose Instant3D, a novel method that generates high-quality and diverse 3D assets from text prompts in a feed-forward manner. We adopt a two-stage paradigm, which first generates a sparse set of four structured and consistent views from text in one shot with a fine-tuned 2D text-to-image diffusion model, and then directly regresses the NeRF from the generated images with a novel transformer-based sparse-view reconstructor. Through extensive experiments, we demonstrate that our method can generate diverse 3D assets of high visual quality within 20 seconds, which is two orders of magnitude faster than previous optimization-based methods that can take 1 to 10 hours.
</details>

[📄 Paper](https://arxiv.org/pdf/2311.06214) | [🌐 Projects Page](https://jiahao.ai/instant3d/)


### Progressive Text-to-3D Generation for Automatic 3D Prototyping

**Author**: Han Yi, Zhedong Zheng, Xiangyu Xu, Tat-seng Chua

**Date**: 26 Sep 2023

<details span>
<summary><b>Abstract</b></summary>
Text-to-3D generation is to craft a 3D object according to a natural language description. This can significantly reduce the workload for manually designing 3D models and provide a more natural way of interaction for users. However, this problem remains challenging in recovering the fine-grained details effectively and optimizing a large-size 3D output efficiently. Inspired by the success of progressive learning, we propose a Multi-Scale Triplane Network (MTN) and a new progressive learning strategy. As the name implies, the Multi-Scale Triplane Network consists of four triplanes transitioning from low to high resolution. The low-resolution triplane could serve as an initial shape for the high-resolution ones, easing the optimization difficulty. To further enable the fine-grained details, we also introduce the progressive learning strategy, which explicitly demands the network to shift its focus of attention from simple coarse-grained patterns to difficult fine-grained patterns. Our experiment verifies that the proposed method performs favorably against existing methods. For even the most challenging descriptions, where most existing methods struggle to produce a viable shape, our proposed method consistently delivers. We aspire for our work to pave the way for automatic 3D prototyping via natural language descriptions.
</details>

[📄 Paper](https://arxiv.org/pdf/2309.14600) | [💻 Code](https://github.com/Texaser/MTN)

## 2022
### Score Jacobian Chaining: Lifting Pretrained 2D Diffusion Models for 3D Generation

**Author**: Haochen Wang, Xiaodan Du, Jiahao Li, Raymond A. Yeh, Greg Shakhnarovich

**Date**: 1 Dec 2022

<details span>
<summary><b>Abstract</b></summary>
A diffusion model learns to predict a vector field of gradients. We propose to apply chain rule on the learned gradients, and back-propagate the score of a diffusion model through the Jacobian of a differentiable renderer, which we instantiate to be a voxel radiance field. This setup aggregates 2D scores at multiple camera viewpoints into a 3D score, and repurposes a pretrained 2D model for 3D data generation. We identify a technical challenge of distribution mismatch that arises in this application, and propose a novel estimation mechanism to resolve it. We run our algorithm on several off-the-shelf diffusion image generative models, including the recently released Stable Diffusion trained on the large-scale LAION dataset.
</details>

[📄 Paper](https://arxiv.org/pdf/2212.00774) | [🌐 Project Page](https://pals.ttic.edu/p/score-jacobian-chaining) | [💻 Code](https://github.com/pals-ttic/sjc/)



### DreamFusion: Text-to-3D using 2D Diffusion

**Author**: Ben Poole, Ajay Jain, Jonathan T. Barron, Ben Mildenhall

**Date**: 29 Sep 2022

<details span>
<summary><b>Abstract</b></summary>
Recent breakthroughs in text-to-image synthesis have been driven by diffusion models trained on billions of image-text pairs. Adapting this approach to 3D synthesis would require large-scale datasets of labeled 3D data and efficient architectures for denoising 3D data, neither of which currently exist. In this work, we circumvent these limitations by using a pretrained 2D text-to-image diffusion model to perform text-to-3D synthesis. We introduce a loss based on probability density distillation that enables the use of a 2D diffusion model as a prior for optimization of a parametric image generator. Using this loss in a DeepDream-like procedure, we optimize a randomly-initialized 3D model (a Neural Radiance Field, or NeRF) via gradient descent such that its 2D renderings from random angles achieve a low loss. The resulting 3D model of the given text can be viewed from any angle, relit by arbitrary illumination, or composited into any 3D environment. Our approach requires no 3D training data and no modifications to the image diffusion model, demonstrating the effectiveness of pretrained image diffusion models as priors.
</details>

[📄 Paper](https://arxiv.org/pdf/2209.14988) | [🌐 Project Page](https://dreamfusion3d.github.io/)
