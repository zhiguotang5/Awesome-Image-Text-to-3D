# Awesome-Image-Text-to-3D [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

A curated list of papers and open-source resources focused on text/image to 3D.

## Table of Contents
- [Dataset of 3D objects](#dataset)

- [Survey on 3D generation](#survey)

- [Open Sourced Unified framework for 3D content generation](#unified-framework-for-3D-content-generation)

- [Single Image to 3D](#single-image-to-3d)

- [Text to 3D](#text-to-3d)

- [Overview of Text-to-3D](#overview-of-text-to-3d)

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

## 2024
### [ECCV '2024] DreamMesh: Jointly Manipulating and Texturing Triangle Meshes for Text-to-3D Generation

**Author**: Haibo Yang, Yang Chen, Yingwei Pan, Ting Yao, Zhineng Chen, Zuxuan Wu, Yu-Gang Jiang, Tao Mei

**Date**: 11 Sep 2024

<details span>
<summary><b>Abstract</b></summary>
Learning radiance fields (NeRF) with powerful 2D diffusion models has garnered popularity for text-to-3D generation. Nevertheless, the implicit 3D representations of NeRF lack explicit modeling of meshes and textures over surfaces, and such surface-undefined way may suffer from the issues, e.g., noisy surfaces with ambiguous texture details or cross-view inconsistency. To alleviate this, we present DreamMesh, a novel text-to-3D architecture that pivots on well-defined surfaces (triangle meshes) to generate high-fidelity explicit 3D model. Technically, DreamMesh capitalizes on a distinctive coarse-to-fine scheme. In the coarse stage, the mesh is first deformed by text-guided Jacobians and then DreamMesh textures the mesh with an interlaced use of 2D diffusion models in a tuning free manner from multiple viewpoints. In the fine stage, DreamMesh jointly manipulates the mesh and refines the texture map, leading to high-quality triangle meshes with high-fidelity textured materials. Extensive experiments demonstrate that DreamMesh significantly outperforms state-of-the-art text-to-3D methods in faithfully generating 3D content with richer textual details and enhanced geometry.
</details>

[📄 Paper](https://arxiv.org/pdf/2409.07454) | [🌐 Project Page](https://dreammesh.github.io/) | [💻 Code (not yet)]()


### MVGaussian: High-Fidelity text-to-3D Content Generation with Multi-View Guidance and Surface Densification

**Author**: Phu Pham, Aradhya N. Mathur, Ojaswa Sharma, Aniket Bera

**Date**: 10 Sep 2024

<details span>
<summary><b>Abstract</b></summary>
The field of text-to-3D content generation has made significant progress in generating realistic 3D objects, with existing methodologies like Score Distillation Sampling (SDS) offering promising guidance. However, these methods often encounter the "Janus" problem-multi-face ambiguities due to imprecise guidance. Additionally, while recent advancements in 3D gaussian splitting have shown its efficacy in representing 3D volumes, optimization of this representation remains largely unexplored. This paper introduces a unified framework for text-to-3D content generation that addresses these critical gaps. Our approach utilizes multi-view guidance to iteratively form the structure of the 3D model, progressively enhancing detail and accuracy. We also introduce a novel densification algorithm that aligns gaussians close to the surface, optimizing the structural integrity and fidelity of the generated models. Extensive experiments validate our approach, demonstrating that it produces high-quality visual outputs with minimal time cost. Notably, our method achieves high-quality results within half an hour of training, offering a substantial efficiency gain over most existing methods, which require hours of training time to achieve comparable results.
</details>

[📄 Paper](https://arxiv.org/pdf/2409.06620) | [🌐 Project Page](https://mvgaussian.github.io/) | [💻 Code](https://github.com/mvgaussian/mvgaussian)


### [NeurIPS '2024] MeshXL: Neural Coordinate Field for Generative 3D Foundation Models

**Author**: Sijin Chen, Xin Chen, Anqi Pang, Xianfang Zeng, Wei Cheng, Yijun Fu, Fukun Yin, Yanru Wang, Zhibin Wang, Chi Zhang, Jingyi Yu, Gang Yu, Bin Fu, Tao Chen

**Date**: 31 May 2024

<details span>
<summary><b>Abstract</b></summary>
The polygon mesh representation of 3D data exhibits great flexibility, fast rendering speed, and storage efficiency, which is widely preferred in various applications. However, given its unstructured graph representation, the direct generation of high-fidelity 3D meshes is challenging. Fortunately, with a pre-defined ordering strategy, 3D meshes can be represented as sequences, and the generation process can be seamlessly treated as an auto-regressive problem. In this paper, we validate the Neural Coordinate Field (NeurCF), an explicit coordinate representation with implicit neural embeddings, is a simple-yet-effective representation for large-scale sequential mesh modeling. After that, we present MeshXL, a family of generative pre-trained auto-regressive models, which addresses the process of 3D mesh generation with modern large language model approaches. Extensive experiments show that MeshXL is able to generate high-quality 3D meshes, and can also serve as foundation models for various down-stream applications.
</details>

[📄 Paper](https://arxiv.org/pdf/2405.20853) | [🌐 Project Page](https://meshxl.github.io/) | [💻 Code](https://github.com/OpenMeshLab/MeshXL)


### CAT3D: Create Anything in 3D with Multi-View Diffusion Models

**Author**: Ruiqi Gao, Aleksander Holynski, Philipp Henzler, Arthur Brussee, Ricardo Martin-Brualla, Pratul Srinivasan, Jonathan T. Barron, Ben Poole

**Date**: 16 May 2024

<details span>
<summary><b>Abstract</b></summary>
Advances in 3D reconstruction have enabled high-quality 3D capture, but require a user to collect hundreds to thousands of images to create a 3D scene. We present CAT3D, a method for creating anything in 3D by simulating this real-world capture process with a multi-view diffusion model. Given any number of input images and a set of target novel viewpoints, our model generates highly consistent novel views of a scene. These generated views can be used as input to robust 3D reconstruction techniques to produce 3D representations that can be rendered from any viewpoint in real-time. CAT3D can create entire 3D scenes in as little as one minute, and outperforms existing methods for single image and few-view 3D scene creation.
</details>

[📄 Paper](https://arxiv.org/pdf/2405.10314) | [🌐 Project Page](https://cat3d.github.io/)


### [CVPR' 2024] ViewDiff: 3D-Consistent Image Generation with Text-to-Image Models

**Author**: Lukas Höllein, Aljaž Božič, Norman Müller, David Novotny, Hung-Yu Tseng, Christian Richardt, Michael Zollhöfer, Matthias Nießner

**Date**: 4 Mar 2024

<details span>
<summary><b>Abstract</b></summary>
3D asset generation is getting massive amounts of attention, inspired by the recent success of text-guided 2D content creation. Existing text-to-3D methods use pretrained text-to-image diffusion models in an optimization problem or fine-tune them on synthetic data, which often results in non-photorealistic 3D objects without backgrounds. In this paper, we present a method that leverages pretrained text-to-image models as a prior, and learn to generate multi-view images in a single denoising process from real-world data. Concretely, we propose to integrate 3D volume-rendering and cross-frame-attention layers into each block of the existing U-Net network of the text-to-image model. Moreover, we design an autoregressive generation that renders more 3D-consistent images at any viewpoint. We train our model on real-world datasets of objects and showcase its capabilities to generate instances with a variety of high-quality shapes and textures in authentic surroundings. Compared to the existing methods, the results generated by our method are consistent, and have favorable visual quality (-30% FID, -37% KID).
</details>

[📄 Paper](https://arxiv.org/pdf/2403.01807) | [🌐 Project Page](https://lukashoel.github.io/ViewDiff/) | [💻 Code](https://github.com/facebookresearch/ViewDiff)


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


### MVDream: Multi-view Diffusion for 3D Generation

**Author**: Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li, Xiao Yang

**Date**:  31 Aug 2023

<details span>
<summary><b>Abstract</b></summary>
We introduce MVDream, a diffusion model that is able to generate consistent multi-view images from a given text prompt. Learning from both 2D and 3D data, a multi-view diffusion model can achieve the generalizability of 2D diffusion models and the consistency of 3D renderings. We demonstrate that such a multi-view diffusion model is implicitly a generalizable 3D prior agnostic to 3D representations. It can be applied to 3D generation via Score Distillation Sampling, significantly enhancing the consistency and stability of existing 2D-lifting methods. It can also learn new concepts from a few 2D examples, akin to DreamBooth, but for 3D generation.
</details>

[📄 Paper](https://arxiv.org/pdf/2308.16512) | [🌐 Projects Page](https://mv-dream.github.io/) | [💻 Code](https://github.com/bytedance/MVDream)


### EfficientDreamer: High-Fidelity and Robust 3D Creation via Orthogonal-view Diffusion Prior

**Author**: Zhipeng Hu, Minda Zhao, Chaoyi Zhao, Xinyue Liang, Lincheng Li, Zeng Zhao, Changjie Fan, Xiaowei Zhou, Xin Yu

**Date**: 25 Aug 2023

<details span>
<summary><b>Abstract</b></summary>
While image diffusion models have made significant progress in text-driven 3D content creation, they often fail to accurately capture the intended meaning of text prompts, especially for view information. This limitation leads to the Janus problem, where multi-faced 3D models are generated under the guidance of such diffusion models. In this paper, we propose a robust high-quality 3D content generation pipeline by exploiting orthogonal-view image guidance. First, we introduce a novel 2D diffusion model that generates an image consisting of four orthogonal-view sub-images based on the given text prompt. Then, the 3D content is created using this diffusion model. Notably, the generated orthogonal-view image provides strong geometric structure priors and thus improves 3D consistency. As a result, it effectively resolves the Janus problem and significantly enhances the quality of 3D content creation. Additionally, we present a 3D synthesis fusion network that can further improve the details of the generated 3D contents. Both quantitative and qualitative evaluations demonstrate that our method surpasses previous text-to-3D techniques.
</details>

[📄 Paper](https://arxiv.org/pdf/2308.13223) | [🌐 Projects Page](https://efficientdreamer.github.io/)


### [ICCV' 2023] Fantasia3D: Disentangling Geometry and Appearance for High-quality Text-to-3D Content Creation

**Author**: Rui Chen, Yongwei Chen, Ningxin Jiao, Kui Jia

**Date**: 24 Mar 2023

<details span>
<summary><b>Abstract</b></summary>
Automatic 3D content creation has achieved rapid progress recently due to the availability of pre-trained, large language models and image diffusion models, forming the emerging topic of text-to-3D content creation. Existing text-to-3D methods commonly use implicit scene representations, which couple the geometry and appearance via volume rendering and are suboptimal in terms of recovering finer geometries and achieving photorealistic rendering; consequently, they are less effective for generating high-quality 3D assets. In this work, we propose a new method of Fantasia3D for high-quality text-to-3D content creation. Key to Fantasia3D is the disentangled modeling and learning of geometry and appearance. For geometry learning, we rely on a hybrid scene representation, and propose to encode surface normal extracted from the representation as the input of the image diffusion model. For appearance modeling, we introduce the spatially varying bidirectional reflectance distribution function (BRDF) into the text-to-3D task, and learn the surface material for photorealistic rendering of the generated surface. Our disentangled framework is more compatible with popular graphics engines, supporting relighting, editing, and physical simulation of the generated 3D assets. We conduct thorough experiments that show the advantages of our method over existing ones under different text-to-3D task settings.
</details>

[📄 Paper](https://arxiv.org/pdf/2303.13873) | [🌐 Projects Page](https://fantasia3d.github.io/) | [💻 Code](https://github.com/Gorilla-Lab-SCUT/Fantasia3D)

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



## Overview of Text-to-3D

This table provides a comprehensive overview of cutting-edge enhanced text-to-3D technologies. Showcasing various methods along with their corresponding projects links, representations, objectives, and motivations. The table contains a plethora of methodologies, each with its unique characteristics and contributions. Motivations driving these advancements vary, ranging from fidelity, efficiency, consistency, controllability, diversity and applicability.

| Method               | Project                                                                                       | Representation             | Optimization              | Directions                           |
|----------------------|-----------------------------------------------------------------------------------------------|-----------------------------|----------------------------|--------------------------------------|
| 3D-CLFusion          | -                                                                                             | NeRF                        | Diffusion+Contrast        | Efficiency                          |
| 3DTopia              | [Link](https://github.com/3DTopia/3DTopia)                                                    | Triplane                    | SDS                        | Efficiency                          |
| 3DFuse               | [Link](https://ku-cvlab.github.io/3DFuse/)                                                    | NeRF                        | SDS/SJC                    | Consistency & Controllability       |
| ATT3D                | [Link](https://research.nvidia.com/labs/toronto-ai/ATT3D/)                                    | NeRF                        | SDS                        | Efficiency                          |
| CLIP-NeRF            | [Link](https://cassiepython.github.io/clipnerf/)                                              | NeRF                        | CLIP                       | Controllability                     |
| Consist3D            | -                                                                                             | SJC                         | SDS+sem+warp+rec          | Consistency                         |
| Consistent3D         | [Link](https://github.com/sail-sg/Consistent3D)                                               | NeRF/DMTet/3DGS             | CDS                        | Consistency                         |
| Control3D            | -                                                                                             | NeRF                        | C-SDS                      | Controllability                     |
| CorrespondentDream   | -                                                                                             | NeRF                        | SDS                        | Fidelity                            |
| CSD                  | [Link](https://xinyu-andy.github.io/Classifier-Score-Distillation)                             | NeRF/DMTet                  | CSD                        | Fidelity                            |
| DATID-3D             | [Link](https://gwang-kim.github.io/datid_3d/)                                                 | Triplane                    | ADA+den                    | Diversity                           |
| Diffusion-SDF        | [Link](https://github.com/ttlmh/Diffusion-SDF)                                                | SDF                         | Diffusion-SDF              | Diversity                           |
| DITTO-NeRF           | [Link](https://janeyeon.github.io/ditto-nerf)                                                 | NeRF                        | inpainting-SDS             | Efficiency, Consistency & Diversity |
| D-SDS                | [Link](https://susunghong.github.io/Debiased-Score-Distillation-Sampling/)                    | NeRF                        | Debiased-SDS               | Consistency                         |
| Dream3D              | [Link](https://bluestyle97.github.io/dream3d/)                                                | DVGO                        | CLIP+prior                 | Controllability                     |
| DreamBooth3D         | [Link](https://dreambooth3d.github.io/)                                                       | NeRF                        | SDS+MVR                    | Controllability                     |
| DreamCraft3D         | [Link](https://github.com/deepseek-ai/DreamCraft3D)                                           | NeuS+DMTet                  | BSD                        | Consistency                         |
| DreamGaussian        | [Link](https://dreamgaussian.github.io/)                                                      | 3DGS                        | SDS                        | Efficiency                          |
| DreamPropeller       | [Link](https://github.com/alexzhou907/DreamPropeller)                                         | NeRF/DMTet                  | SDS/VSD                    | Efficiency                          |
| DreamTime            | -                                                                                             | NeRF                        | TP-VSD/TP-SDS             | Fidelity & Diversity                |
| Dreamer XL           | [Link](https://github.com/xingy038/Dreamer-XL)                                                | 3DGS                        | TSM                        | Consistency                         |
| EfficientDreamer     | [Link](https://efficientdreamer.github.io/)                                                   | NeuS+DMTet                  | SDS/VDS                    | Consistency                         |
| ExactDreamer         | [Link](https://github.com/zymvszym/ExactDreamer)                                              | 3DGS                        | ESM                        | Fidelity & Consistency              |
| Fantasia3D           | [Link](https://fantasia3d.github.io/)                                                         | DMTet                       | SDS                        | Fidelity                            |
| FSD                  | -                                                                                             | NeRF                        | FSD                        | Diversity                           |
| GaussianDiffusion    | -                                                                                             | 3DGS                        | SDS                        | Fidelity & Consistency              |
| GaussianDreamer      | [Link](https://taoranyi.com/gaussiandreamer/)                                                 | 3DGS                        | SDS                        | Efficiency                          |
| GSGEN                | [Link](https://gsgen3d.github.io/)                                                            | 3DGS                        | SDS                        | Efficiency & Consistency            |
| Grounded-Dreamer     | -                                                                                             | NeRF                        | SDS                        | Fidelity                            |
| HD-Fusion            | -                                                                                             | SDF+DMTet                   | SDS/VSD                    | Fidelity                            |
| HiFA                 | [Link](https://hifa-team.github.io/HiFA-site/)                                                | NeRF                        | SDS                        | Fidelity & Consistency              |
| InNeRF360            | [Link](https://ivrl.github.io/InNeRF360/)                                                     | NeRF                        | DSDS                       | Consistency & Controllability       |
| Instant3D            | [Link](https://jiahao.ai/instant3d/)                                                          | Triplane                    | MSE+LPIPS                  | Efficiency & Diversity              |
| Interactive3D        | [Link](https://interactive-3d.github.io/)                                                     | 3DGS+InstantNGP             | interactive-SDS            | Controllability                     |
| IT3D                 | [Link](https://github.com/buaacyw/IT3D-text-to-3D)                                            | NeRF+Mesh                   | SDS                        | Fidelity & Consistency              |
| LI3D                 | -                                                                                             | NeRF                        | SDS                        | Controllability                     |
| LucidDreamer         | [Link](https://github.com/EnVision-Research/LucidDreamer)                                     | 3DGS                        | ISM                        | Fidelity                            |
| Magic3D              | [Link](https://research.nvidia.com/labs/dir/magic3d)                                          | NeRF+DMTet                  | SDS                        | Fidelity & Efficiency               |
| MATLABER             | [Link](https://sheldontsui.github.io/projects/Matlaber)                                       | DMTet                       | SDS                        | Fidelity                            |
| MTN                  | -                                                                                             | Multi-Scale Triplane        | SDS                        | Fidelity                            |
| MVControl            | [Link](https://github.com/WU-CVGL/MVControl)                                                  | NeuS/DMTet                  | SDS                        | Controllability                     |
| MVDream              | [Link](https://mv-dream.github.io/)                                                           | NeRF                        | SDS                        | Consistency                         |
| Perp-Neg             | [Link](https://perp-neg.github.io/)                                                           | NeRF                        | SDS                        | Consistency                         |
| PI3D                 | -                                                                                             | Triplane                    | SDS                        | Efficiency & Consistency            |
| Points-to-3D         | -                                                                                             | NeRF                        | SDS                        | Consistency & Controllability       |
| ProlificDreamer      | [Link](https://ml.cs.tsinghua.edu.cn/prolificdreamer/)                                        | NeRF                        | VSD                        | Fidelity & Diversity                |
| RichDreamer          | [Link](https://aigc3d.github.io/richdreamer/)                                                 | NeRF/DMTet                  | SDS                        | Consistency                         |
| Sherpa3D             | [Link](https://liuff19.github.io/Sherpa3D/)                                                   | DMTet                       | SDS                        | Consistency                         |
| SweetDreamer         | [Link](https://sweetdreamer3d.github.io/)                                                     | NeRF/DMTet                  | SDS                        | Consistency                         |
| TAPS3D               | [Link](https://github.com/plusmultiply/TAPS3D)                                                | DMTet                       | CLIP+IMG                   | Fidelity & Diversity                |
| TextMesh             | [Link](https://fabi92.github.io/textmesh/)                                                    | SDF+Mesh                    | SDS                        | Fidelity                            |
| X-Dreamer            | [Link](https://xmu-xiaoma666.github.io/Projects/X-Dreamer)                                    | DMTet                       | SDS+AMA                    | Fidelity                            |
| X-Oscar              | [Link](https://xmu-xiaoma666.github.io/Projects/X-Oscar/)                                     | SMPL-X                      | ASDS                       | Fidelity                            |
