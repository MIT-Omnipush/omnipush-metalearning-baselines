# OmniPush meta-learning baselines
Baselines for the meta-learning benchmark of the [OmniPush dataset](http://lis.csail.mit.edu/pubs/alet-bauza-iros2019.pdf).

You can find the data at ftp://omnipush.mit.edu/omnipush/meta-learning_data/
## Pooling all the data
Simple baseline that aggregates all datasets into one.
## Attentive Neural Processes baseline
Implementation of [Attentive Neural Processes](https://arxiv.org/abs/1901.05761) by Kim et al.

Code mainly based on:
  * https://github.com/soobinseo/Attentive-Neural-Process and a couple modifications based on
  * https://github.com/deepmind/neural-processes

## Comments or questions
  For comments, questions or suggestions please email me at alet at mit dot edu.

## License and citation
This code is provided under MIT license; if you used the code from this repo, consider citing the following papers.
```
@article{bauza2019omnipush,
  title={Omnipush: accurate, diverse, real-world dataset of pushing dynamics with RGB-D video},
  author={Bauza, Maria and Alet, Ferran and Lin, Yen-Chen and Lozano-Perez, Tomas and Kaelbling, Leslie P and Isola, Phillip and Rodriguez, Alberto},
  journal={arXiv preprint arXiv:1910.00618},
  year={2019}
}
@article{kim2019attentive,
  title={Attentive neural processes},
  author={Kim, Hyunjik and Mnih, Andriy and Schwarz, Jonathan and Garnelo, Marta and Eslami, Ali and Rosenbaum, Dan and Vinyals, Oriol and Teh, Yee Whye},
  journal={arXiv preprint arXiv:1901.05761},
  year={2019}
}
```
