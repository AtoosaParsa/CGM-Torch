# CGM-Torch: A Differentiable Simulator for Granular Materials

This repository contains the source code for all the experiments in the following paper:

[Parsa, A., O'Hern, C. S., Kramer-Bottiglio, R., & Bongard, J. (2024). Gradient-based Design of Computational Granular Crystals. arXiv preprint arXiv:2404.04825.](https://arxiv.org/abs/2404.04825)

</br>

<p align="center">
  <img src="https://github.com/AtoosaParsa/CGM-Torch/blob/main/media/overview.png"  width="700">
</p>

</br>
</br>

## Installation
Clone this repository and install the following using your preferred python environment or package managment tool:

## Usage
### Training a new model:
```
python train.py --name "test" --savedir "./test/" --seed 1
```

### Loading a previously trained model:
```
python loadModel.py --savedir "./test/" --name "test" --seed 1 --plotName 'AND'
```

## AND Gate
![](https://github.com/AtoosaParsa/GCTorch/blob/main/media/AND_config.gif)
![](https://github.com/AtoosaParsa/GCTorch/blob/main/media/AND_plot.gif)

## XOR Gate
![](https://github.com/AtoosaParsa/GCTorch/blob/main/media/XOR_config.gif)
![](https://github.com/AtoosaParsa/GCTorch/blob/main/media/XOR_plot.gif)

## Citation
If you find our paper or this repository useful or relevant to your work please consider citing us:

```
@article{parsa2024gradient,
  title={Gradient-based Design of Computational Granular Crystals},
  author={Parsa, Atoosa and O'Hern, Corey S and Kramer-Bottiglio, Rebecca and Bongard, Josh},
  journal={arXiv preprint arXiv:2404.04825},
  year={2024}
}
```
