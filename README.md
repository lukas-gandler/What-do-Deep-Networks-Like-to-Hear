# What do Deep Networks Like to Hear?
This repository accompanies my master thesis with the same title from the [Institute for Computational Perception](https://www.jku.at/en/institute-of-computational-perception/) at Johannes Kepler University in Linz.  

# Abstract
Advances in explainable artificial intelligence techniques shine light on how the inner workings of neural networks function. The gained insights also help to further architectural designs and mitigate flaws or vulnerabilities. Past works employed autoencoder to investigate the preferences of [image](https://arxiv.org/abs/1803.08337) and [sentence classification](https://arxiv.org/abs/1909.04547) networks at their input level. In this thesis, this idea is extended to environmental sound classification. To this end, audio waveforms are passed through a [1D convolutional autoencoder](https://github.com/archinetai/archisound) and the resulting reconstructions are passed to the classifier networks to make predictions. The prediction error is then backpropagated to only fine-tune the weights of the autoencoder. The weights of the classification network stay fixed. 

For this thesis, three architectures are considered: The normal [MobileNet-V3](https://github.com/fschmid56/EfficientAT), a changed variant of the MobileNet-V3 architecture that introduces dynamic attention layers called [Dynamic MobileNet](https://github.com/fschmid56/EfficientAT) and the vision transformer [PaSST](https://github.com/kkoutini/PaSST/tree/main). These models were selected due to their strong performance in environmental sound classification and architectural differences. All experiments are conducted on the [ESC-50](https://github.com/karolpiczak/ESC-50) dataset which consists of five second environmental sound audio samples categorized into 50 classes. 

Investigations of the learned reconstruction transformations show differences in how the three classification architectures perceive their inputs. Especially Dynamic MobileNet differs the most from the other two. Ablation studies are conducted to investigate which alterations to the MobileNet architecture caused these changes in behaviour. The results also show indications that lower frequencies hold the most information for environmental sound classification.


# References
[1] Sebastian Palacio, Joachim Folz, Jörn Hees, Federico Raue, Damian Borth,
 and Andreas Dengel. 2018. What do deep networks like to see? In Proceed
ings of the IEEE Conference on Computer Vision and Pattern Recognition,
 pp. 3108–3117.

[2] Jonas Pfeiffer, Aishwarya Kamath, Iryna Gurevych, and Sebastian
 Ruder. 2019. What do Deep Networks Like to Read? arXiv preprint
 arXiv:1909.04547.

[3] Flavio Schneider. 2023. Archisound: Audio generation with diffusion. arXiv
 preprint arXiv:2301.13267.

[4] Florian Schmid, Khaled Koutini, and Gerhard Widmer. 2023. Efficient
 large-scale audio tagging via transformer-to-cnn knowledge distillation. In
 ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech
 and Signal Processing (ICASSP). IEEE, pp. 1–5.

[5] Florian Schmid, Khaled Koutini, and Gerhard Widmer. 2024. Dynamic
 Convolutional Neural Networks as Efficient Pre-trained Audio Models.
 IEEE/ACM Transactions on Audio, Speech, and Language Processing.

[6] Khaled Koutini, Jan Schlüter, Hamid Eghbal-zadeh, and Gerhard Widmer, “Efficient Training of Audio Transformers with Patchout,” in Interspeech, 2022.

[7] Karol J. Piczak. 2015. ESC: Dataset for Environmental Sound Classifica
tion. In Proceedings of the 23rd Annual ACM Conference on Multimedia.
 ACM Press, Brisbane, Australia, (October 13, 2015), pp. 1015–1018. isbn:
 978-1-4503-3459-4. doi: 10.1145/2733373.2806390. http://dl.acm.org/citat
 ion.cfm?doid=2733373.2806390. 
