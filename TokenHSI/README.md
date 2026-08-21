<p align="center">
<h1 align="center"<strong>TokenHSI: Unified Synthesis of Physical Human-Scene Interactions through Task Tokenization</strong></h1>
  <p align="center">
    <a href='https://liangpan99.github.io/' target='_blank'>Liang Pan</a><sup>1,2</sup>
    ·
    <a href='https://zeshiyang.github.io/' target='_blank'>Zeshi Yang</a> <sup>3</sup>
    ·
    <a href='https://frank-zy-dou.github.io/' target='_blank'>Zhiyang Dou</a><sup>2</sup>
    ·
    <a href='https://wenjiawang0312.github.io/' target='_blank'>Wenjia Wang</a><sup>2</sup>
    ·
    <a href='https://www.buzhenhuang.com/about/' target='_blank'>Buzhen Huang</a><sup>4</sup>
    ·
    <a href='https://scholar.google.com/citations?user=KNWTvgEAAAAJ&hl=en' target='_blank'>Bo Dai</a><sup>2,5</sup>
    ·
    <a href='https://i.cs.hku.hk/~taku/' target='_blank'>Taku Komura</a><sup>2</sup>
    ·
    <a href='https://scholar.google.com/citations?user=GStTsxAAAAAJ&hl=en&oi=ao' target='_blank'>Jingbo Wang</a><sup>1</sup>
    <br>
    <sup>1</sup>Shanghai AI Lab <sup>2</sup>The University of Hong Kong <sup>3</sup>Independent Researcher <sup>4</sup>Southeast University <sup>5</sup>Feeling AI
    <br>
    <strong>CVPR 2025</strong>
    <br>
    <strong>🏆️ Oral Presentation (Top 3.3%)</strong>
    <br>
    Also <strong>Spotlight</strong> in the 1st Workshop on Humanoid Agents at CVPR 2025
  </p>
</p>
<p align="center">
  <a href='https://arxiv.org/abs/2503.19901'>
    <img src='https://img.shields.io/badge/arXiv-2503.19901-A42C25?style=flat&logo=arXiv&logoColor=A42C25'></a>
  <a href='https://arxiv.org/pdf/2503.19901'>
    <img src='https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow'></a>
  <a href='https://liangpan99.github.io/TokenHSI/'>
    <img src='https://img.shields.io/badge/Project-Page-green?style=flat&logo=Google%20chrome&logoColor=green'></a>
</p>

## 🏠 About
<div style="text-align: center;">
    <img src="https://github.com/liangpan99/TokenHSI/blob/page/static/images/teaser.png" width=100% >
</div>
Introducing TokenHSI, a unified model that enables physics-based characters to perform diverse human-scene interaction tasks. It excels at seamlessly unifying multiple <b>foundational HSI skills</b> within a single transformer network and flexibly adapting learned skills to <b>challenging new tasks</b>, including skill composition, object/terrain shape variation, and long-horizon task completion.
</br>

## 📹 Demo
<p align="center">
    <img src="assets/longterm_demo_isaacgym.gif" align="center" width=60% >
    <br>
    Long-horizon Task Completion in a Complex Dynamic Environment
</p>

<!-- ## 🕹 Pipeline
<div style="text-align: center;">
    <img src="https://github.com/liangpan99/TokenHSI/blob/page/static/images/pipeline.jpg" width=100% >
</div> -->

## 🔥 News  
- **[2025-04-07]** <b>Released full code. Please note to download the latest datasets and models from Hugging Face.</b>
- **[2025-04-06]** Released three skill composition tasks with pre-trained models.
- **[2025-04-05]** TokenHSI has been selected as an oral paper at CVPR 2025! 🎉
- **[2025-04-03]** Released long-horizon task completion with a pre-trained model.
- **[2025-04-01]** We just updated the Getting Started section. You can play TokenHSI now!
- **[2025-03-31]** We've released the codebase and checkpoint for the foundational skill learning part.

## 📝 TODO List  
- [x] Release foundational skill learning 
- [x] Release policy adaptation - skill composition  
- [x] Release policy adaptation - object shape variation
- [x] Release policy adaptation - terrain shape variation
- [x] Release policy adaptation - long-horizon task completion

## 📖 Getting Started

### Dependencies

Follow the following instructions: 

1. Create new conda environment and install pytroch

    ```
    conda create -n tokenhsi python=3.8
    conda activate tokenhsi
    conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
    pip install -r requirements.txt
    ```

2. Install [IsaacGym Preview 4](https://developer.nvidia.com/isaac-gym) 

    ```
    cd IsaacGym_Preview_4_Package/isaacgym/python
    pip install -e .

    # add your conda env path to ~/.bashrc
    export LD_LIBRARY_PATH="your_conda_env_path/lib:$LD_LIBRARY_PATH"
    ```

3. Install pytorch3d (optional, if you want to run the long-horizon task completion demo)

    **We use pytorch3d to rapidly render height maps of dynamic objects for thousands of simulation environments.**

    ```
    conda install -c fvcore -c iopath -c conda-forge fvcore iopath
    pip install git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7
    ```

4. Download [SMPL body models](https://smpl.is.tue.mpg.de/) and organize them as follows:

    ```
    |-- assets
    |-- body_models
        |-- smpl
            |-- SMPL_FEMALE.pkl
            |-- SMPL_MALE.pkl
            |-- SMPL_NEUTRAL.pkl
            |-- ...
    |-- lpanlib
    |-- tokenhsi
    ```

### Motion & Object Data

We provide two methods to generate the motion and object data.

* Download pre-processed data from [Hugging Face](https://huggingface.co/datasets/lianganimation/TokenHSI). Please follow the instruction in the dataset page.

* Generate data from source:

  1. Download [AMASS (SMPL-X Neutral)](https://amass.is.tue.mpg.de/), [SAMP](https://samp.is.tue.mpg.de/), and [OMOMO](https://github.com/lijiaman/omomo_release).

  2. Modify dataset paths in ```tokenhsi/data/dataset_cfg.yaml``` file.

      ```
      # Motion datasets, please use your own paths
      amass_dir: "/YOUR_PATH/datasets/AMASS"
      samp_pkl_dir: "/YOUR_PATH/datasets/samp"
      omomo_dir: "/YOUR_PATH/datasets/OMOMO/data"
      ```

  3. We still need to download the pre-processed data from [Hugging Face](https://huggingface.co/datasets/lianganimation/TokenHSI). But now we only require the object data.

  4. Run the following script:

      ```
      bash tokenhsi/scripts/gen_data.sh
      ```

### Checkpoints

Download checkpoints from [Hugging Face](https://huggingface.co/lianganimation/TokenHSI). Please follow the instruction in the model page.

## 🕹️ Play TokenHSI!

* Single task policy trained with AMP

  * Path-following

      ```
      # test
      sh tokenhsi/scripts/single_task/traj_test.sh
      # train
      sh tokenhsi/scripts/single_task/traj_train.sh
      ```

  * Sitting

      ```
      # test
      sh tokenhsi/scripts/single_task/sit_test.sh
      # train
      sh tokenhsi/scripts/single_task/sit_train.sh
      ```
  * Climbing

      ```
      # test
      sh tokenhsi/scripts/single_task/climb_test.sh
      # train
      sh tokenhsi/scripts/single_task/climb_train.sh
      ```

  * Carrying

      ```
      # test
      sh tokenhsi/scripts/single_task/carry_test.sh
      # train
      sh tokenhsi/scripts/single_task/carry_train.sh
      ```

* TokenHSI's unified transformer policy

  * Foundational Skill Learning

      ```
      # test
      sh tokenhsi/scripts/tokenhsi/stage1_test.sh
      # eval
      sh tokenhsi/scripts/tokenhsi/stage1_eval.sh carry # we need to specify a task to eval, e.g., traj, sit, climb, or carry.
      # train
      sh tokenhsi/scripts/tokenhsi/stage1_train.sh
      ```

      If you successfully run the test command, you will see:
      <p align="center">
        <img src="assets/stage1_demo.gif" align="center" width=60% >
      </p>


  * Policy Adaptation - Skill Composition
    * Traj + Carry

      ```
      # test
      sh tokenhsi/scripts/tokenhsi/stage2_comp_traj_carry_test.sh
      # eval
      sh tokenhsi/scripts/tokenhsi/stage2_comp_traj_carry_eval.sh
      # train
      sh tokenhsi/scripts/tokenhsi/stage2_comp_traj_carry_train.sh
      ```

      If you successfully run the test command, you will see:
    <p align="center">
      <img src="assets/stage2_comp_traj_carry.gif" align="center" width=60% >
    </p>

    * Sit + Carry

      ```
      # test
      sh tokenhsi/scripts/tokenhsi/stage2_comp_sit_carry_test.sh
      # eval
      sh tokenhsi/scripts/tokenhsi/stage2_comp_sit_carry_eval.sh
      # train
      sh tokenhsi/scripts/tokenhsi/stage2_comp_sit_carry_train.sh
      ```

      If you successfully run the test command, you will see:
    <p align="center">
      <img src="assets/stage2_comp_sit_carry.gif" align="center" width=60% >
    </p>

    * Climb + Carry

      ```
      # test
      sh tokenhsi/scripts/tokenhsi/stage2_comp_climb_carry_test.sh
      # eval
      sh tokenhsi/scripts/tokenhsi/stage2_comp_climb_carry_eval.sh
      # train
      sh tokenhsi/scripts/tokenhsi/stage2_comp_climb_carry_train.sh
      ```

      If you successfully run the test command, you will see:
    <p align="center">
      <img src="assets/stage2_comp_climb_carry.gif" align="center" width=60% >
    </p>


  * Policy Adaptation - Object Shape Variation

    * Carrying: Box-2-Chair

      ```
      # test
      sh tokenhsi/scripts/tokenhsi/stage2_object_chair_test.sh
      # eval
      sh tokenhsi/scripts/tokenhsi/stage2_object_chair_eval.sh
      # train
      sh tokenhsi/scripts/tokenhsi/stage2_object_chair_train.sh
      ```

      If you successfully run the test command, you will see:
    <p align="center">
      <img src="assets/stage2_object_chair.gif" align="center" width=60% >
    </p>

    * Carrying: Box-2-Table

      ```
      # test
      sh tokenhsi/scripts/tokenhsi/stage2_object_table_test.sh
      # eval
      sh tokenhsi/scripts/tokenhsi/stage2_object_table_eval.sh
      # train
      sh tokenhsi/scripts/tokenhsi/stage2_object_table_train.sh
      ```

      If you successfully run the test command, you will see:
    <p align="center">
      <img src="assets/stage2_object_table.gif" align="center" width=60% >
    </p>

  * Policy Adaptation - Terrain Shape Variation

    * Path-following

      ```
      # test
      sh tokenhsi/scripts/tokenhsi/stage2_terrain_traj_test.sh
      # eval
      sh tokenhsi/scripts/tokenhsi/stage2_terrain_traj_eval.sh
      # train
      sh tokenhsi/scripts/tokenhsi/stage2_terrain_traj_train.sh
      ```

      If you successfully run the test command, you will see:
    <p align="center">
      <img src="assets/stage2_terrain_traj.gif" align="center" width=60% >
    </p>
  
  * Carrying

      ```
      # test
      sh tokenhsi/scripts/tokenhsi/stage2_terrain_carry_test.sh
      # eval
      sh tokenhsi/scripts/tokenhsi/stage2_terrain_carry_eval.sh
      # train
      sh tokenhsi/scripts/tokenhsi/stage2_terrain_carry_train.sh
      ```

      If you successfully run the test command, you will see:
    <p align="center">
      <img src="assets/stage2_terrain_carry.gif" align="center" width=60% >
    </p>

  * Policy Adaptation - Long-horizon Task Completion

      ```
      # test
      sh tokenhsi/scripts/tokenhsi/stage2_longterm_test.sh
      # eval
      sh tokenhsi/scripts/tokenhsi/stage2_longterm_eval.sh
      # train
      sh tokenhsi/scripts/tokenhsi/stage2_longterm_train.sh
      ```

### Viewer Shortcuts

| Keyboard | Function |
| ---- | --- |
| F | focus on humanoid |
| Right Click + WASD | change view port |
| Shift + Right Click + WASD | change view port fast |
| K | visualize lines |
| L | record screenshot, press again to stop recording|

The recorded screenshots are saved in ``` output/imgs/ ```. You can use ``` lpanlib/others/video.py ``` to generate mp4 video from the recorded images.

```
python lpanlib/others/video.py --imgs_dir output/imgs/example_path --delete_imgs
```

## 🔗 Citation

If you find our work helpful, please cite:

```bibtex
@inproceedings{pan2025tokenhsi,
  title={TokenHSI: Unified Synthesis of Physical Human-Scene Interactions through Task Tokenization},
  author={Pan, Liang and Yang, Zeshi and Dou, Zhiyang and Wang, Wenjia and Huang, Buzhen and Dai, Bo and Komura, Taku and Wang, Jingbo},
  booktitle={CVPR},
  year={2025},
}

@inproceedings{pan2024synthesizing,
  title={Synthesizing physically plausible human motions in 3d scenes},
  author={Pan, Liang and Wang, Jingbo and Huang, Buzhen and Zhang, Junyu and Wang, Haofan and Tang, Xu and Wang, Yangang},
  booktitle={2024 International Conference on 3D Vision (3DV)},
  pages={1498--1507},
  year={2024},
  organization={IEEE}
}
```

Please also consider citing the following papers that inspired TokenHSI.

```bibtex
@article{tessler2024maskedmimic,
  title={Maskedmimic: Unified physics-based character control through masked motion inpainting},
  author={Tessler, Chen and Guo, Yunrong and Nabati, Ofir and Chechik, Gal and Peng, Xue Bin},
  journal={ACM Transactions on Graphics (TOG)},
  volume={43},
  number={6},
  pages={1--21},
  year={2024},
  publisher={ACM New York, NY, USA}
}

@article{he2024hover,
  title={Hover: Versatile neural whole-body controller for humanoid robots},
  author={He, Tairan and Xiao, Wenli and Lin, Toru and Luo, Zhengyi and Xu, Zhenjia and Jiang, Zhenyu and Kautz, Jan and Liu, Changliu and Shi, Guanya and Wang, Xiaolong and others},
  journal={arXiv preprint arXiv:2410.21229},
  year={2024}
}
```

## 👏 Acknowledgements and 📚 License

This repository builds upon the following awesome open-source projects:

- [ASE](https://github.com/nv-tlabs/ASE): Contributes to the physics-based character control codebase  
- [Pacer](https://github.com/nv-tlabs/pacer): Contributes to the procedural terrain generation and trajectory following task
- [rl_games](https://github.com/Denys88/rl_games): Contributes to the reinforcement learning code  
- [OMOMO](https://github.com/lijiaman/omomo_release)/[SAMP](https://samp.is.tue.mpg.de/)/[AMASS](https://amass.is.tue.mpg.de/)/[3D-Front](https://arxiv.org/abs/2011.09127): Used for the reference dataset construction
- [InterMimic](https://github.com/Sirui-Xu/InterMimic): Used for the github repo readme design 

This codebase is released under the [MIT License](LICENSE).  
Please note that it also relies on external libraries and datasets, each of which may be subject to their own licenses and terms of use.

## 🌟 Star History

<p align="center">
    <a href="https://www.star-history.com/#liangpan99/TokenHSI&Date" target="_blank">
        <img width="500" src="https://api.star-history.com/svg?repos=liangpan99/TokenHSI&type=Date" alt="Star History Chart">
    </a>
<p>

