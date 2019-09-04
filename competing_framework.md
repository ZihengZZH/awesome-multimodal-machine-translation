## Comparison of competing frameworks

According to the category introduced in [the introduction slides by Barrault](https://www.clsp.jhu.edu/wp-content/uploads/sites/75/2018/06/2018-06-22-Barrault-Multimodal-MT.pdf), the competing frameworks in multimodal machine translation are categorized into the following subgroups:
1. Multimodal Attention Mechanism 
2. Integration of Visual Information
3. Multitask Learning
4. Visual Pivot

> En2Fr and Fr2En were introduced in the shared task of WMT'18, thus previous framework could not produce results on such bilingual language pair.

### Multimodal Attention Mechanism

| Authors | Paper | BLEU (EnDe) | METEOR (En-De) | BLEU (EnFr) | METEOR (En-Fr) | Links |
| --      | --    | :--:        | :--:           | :--:        | :--:           | --    |
| Caglayan et al. 2016 | Does Multimodality Help Human and Machine for Translation and Image Captioning? | 19.2 | 32.3 | - | - | [[pdf](https://arxiv.org/pdf/1605.09186.pdf)] |
| Caglayan et al. 2016 | Multimodal Attention for Neural Machine Translation | 19.7 | 35.1 | - | - | [[pdf](https://arxiv.org/pdf/1609.03976.pdf)] |
| Delbrouck et al. 2017 | Multimodal Compact Bilinear Pooling for Multimodal Neural Machine Translation | 29.7 | 48.8 | - | - | [[pdf](https://arxiv.org/pdf/1703.08084.pdf)] |
| Libovicky et al. 2017 | Attention Strategies for Multi-Source Sequence-to-Sequence Learning | 32.1 | 49.1 | - | - | [[pdf](https://aclweb.org/anthology/P17-2031)] |
| Caglayan et al. 2018 | LIUM-CVC Submissions for WMT18 Multimodal Translation Task | 31.4 | 51.4 | 39.5 | 59.9 | [[pdf](http://statmt.org/wmt18/pdf/WMT065.pdf)] |
| Helcl et al. 2018 | CUNI System for the WMT18 Multimodal Translation Task | 32.5 | 52.3 | 40.6 | 61.0 | [[pdf](http://statmt.org/wmt18/pdf/WMT068.pdf)] | 
| Zhou et al. 2018 | A Visual Attention Grounding Neural Model for Multimodal Machine Translation | *63.5*\* | *65.7*\* | *65.8*\* | *68.9*\* | [[pdf](https://www.aclweb.org/anthology/D18-1400)] |
| Caglayan et al. 2019| Probing the Need for Visual Context in Multimodal Machine Translation | - | - | - |68.8 | [[pdf](https://arxiv.org/pdf/1903.08678.pdf)] |
| Su et al. 2019 | Unsupervised Multi-modal Neural Machine Translation | 25.0\* | - | 40.1\* | - | [[pdf](https://arxiv.org/pdf/1811.11365.pdf)] |
| Ive et al. 2019 | Distilling Translations with Visual Awareness | 27.7 | 46.5 | 37.8 | 57.2 | [[pdf](https://arxiv.org/pdf/1906.07701.pdf)] |
| Hirasawa et al. 2019 | Debiasing Word Embedding Improves Multimodal Machine Translation | 36.4\* | 55.2\* | 58.5\* | 73.6\* | [[pdf](https://arxiv.org/pdf/1905.10464.pdf)] |

> The dataset used in the evaluation is assumed Multi30K, unless indicated. Furthermore, the framework is generally evaluated on the year's MWT shared task, e.g. 2018 framework on WMT'18. Only the best results are recorded, and more comprehensive results refer to original paper.

> Zhou et al. 2018 experimented their models on IKEA dataset. Su et al. 2019 reported their experimental results in En-Fn and En-De separately, whose unweighted averages are shown in the table. Hirasawa et al. 2019 reported their results on uni-directional translation tasks: En2Ge and En2Fr.

### Integration of Visual information

| Authors | Paper | BLEU (EnDe) | METEOR (En-De) | BLEU (EnFr) | METEOR (En-Fr) | Links |
| --      | --    | :--:        | :--:           | :--:        | :--:           | --    |
| Huang et al. 2016 | Attention-based Multimodal Neural Machine Translation | 36.5 | 54.1 | - | - | [[pdf](https://www.aclweb.org/anthology/W16-2360)] |
| Lala et al. 2017 | Unraveling the Contribution of Image Captioning and Neural Machine Translation for Multimodal Machine Translation | 39.1 | 36.8 | - | - | [[pdf](https://ufal.mff.cuni.cz/pbml/108/art-lala-madhyastha-wang-specia.pdf)] | 
| Calixto et al. 2017 | Doubly-Attentive Decoder for Multi-modal Neural Machine Translation | 39.0 | 56.8 | - | - | [[pdf](https://arxiv.org/pdf/1702.01287.pdf)] [[github](https://github.com/iacercalixto/MultimodalNMT)] |
| Calixto et al. 2017 | Incorporating Global Visual Features into Attention-Based Neural Machine Translation | 41.3\* | 59.2\* | - | - | [[pdf](https://www.aclweb.org/anthology/D17-1105)] |
| Gronroos et al. 2018 | The MeMAD Submission to the WMT18 Multimodal Translation Task | 38.5 | 56.6 | 44.1 | 64.3 | [[pdf](http://statmt.org/wmt18/pdf/WMT066.pdf)] |
| Lala et al. 2018 | Sheffield Submissions for WMT18 Multimodal Translation Shared Task | 30.5 | 50.7 | 38.8 | 59.8 | [[pdf](http://statmt.org/wmt18/pdf/WMT069.pdf)] |
| Zheng et al. 2018 | Ensemble Sequence Level Training for Multimodal MT: OSU-Baidu WMT18 Multimodal Translation System Report | 32.3 | 50.9 | 39.0 | 59.5 | [[pdf](http://statmt.org/wmt18/pdf/WMT070.pdf)] |
| Delbrouck et al. 2018 | UMONS Submission for WMT18 Multimodal Translation Task | 31.1 | 51.6 | 39.4 | 60.1 | [[pdf](http://statmt.org/wmt18/pdf/WMT071.pdf)] |
| Caglayan et al. 2019| Probing the Need for Visual Context in Multimodal Machine Translation | - | - | - | 68.9 | [[pdf](https://arxiv.org/pdf/1903.08678.pdf)] |
| Calixto et al. 2019 | Latent Variable Model for Multi-modal Translation | 30.1 | 49.9 | - | - | [[pdf](https://www.aclweb.org/anthology/P19-1642)] |
| Hirasawa et al. 2019 | Debiasing Word Embedding Improves Multimodal Machine Translation | 34.8\* | 53.9\* | 56.3\* | 72.2\* | [[pdf](https://arxiv.org/pdf/1905.10464.pdf)] |

> The dataset used in the evaluation is assumed Multi30K, unless indicated. Furthermore, the framework is generally evaluated on the year's MWT shared task, e.g. 2018 framework on WMT'18. Only the best results are recorded, and more comprehensive results refer to original paper.

> Hirasawa et al. 2019 reported their results on uni-directional translation tasks: En2Ge and En2Fr. Calixto et al. 2017 reported their results on uni-directional translation task: En2De.

### Multitask Learning

| Authors | Paper | BLEU (EnDe) | METEOR (En-De) | BLEU (EnFr) | METEOR (En-Fr) | Links |
| --      | --    | :--:        | :--:           | :--:        | :--:           | --    |
| Elliott et al. 2017 | Imagination improves Multimodal Translation | 36.8\* | 55.8\* | - | - | [[pdf](https://arxiv.org/pdf/1705.04350.pdf)] 
| Helcl et al. 2018 | CUNI System for the WMT18 Multimodal Translation Task | 30.2 | 51.7 | 40.4 | 60.7 | [[pdf](http://statmt.org/wmt18/pdf/WMT068.pdf)] | 
| Hirasawa et al. 2019 | Debiasing Word Embedding Improves Multimodal Machine Translation | 36.6\* | 55.4\* | 58.1\* | 73.2\* | [[pdf](https://arxiv.org/pdf/1905.10464.pdf)] |

> The dataset used in the evaluation is assumed Multi30K, unless indicated. Furthermore, the framework is generally evaluated on the year's MWT shared task, e.g. 2018 framework on WMT'18. Only the best results are recorded, and more comprehensive results refer to original paper.

> Elliott et al. 2017 reported their translation results only on Multi30K En2De. Hirasawa et al. 2019 reported their results on uni-directional translation tasks: En2Ge and En2Fr.

### Visual Pivot

| Authors | Paper | BLEU (EnDe) | METEOR (En-De) | BLEU (EnFr) | METEOR (En-Fr) | Links |
| --      | --    | :--:        | :--:           | :--:        | :--:           | --    |
| Nakayama et al. 2017 | Zero-resource Machine Translation by Multimodal Encoder-decoder Network with Multimedia Pivot | 13.8\* | - | - | - | [[pdf](https://arxiv.org/pdf/1611.04503.pdf)] |
| Gwinnup et al. 2018 | The AFRL-Ohio State WMT18 Multimodal System: Combining Visual with Traditional | 24.3 | 45.4 | - | - | [[pdf](http://statmt.org/wmt18/pdf/WMT067.pdf)] |
| Chen et al. 2019 | From Words to Sentences: A Progressive Learning Approach for Zero-resource Machine Translation with Visual Pivots | 20.6\* | - | - | - | [[pdf](https://arxiv.org/pdf/1906.00872.pdf)] |

> The dataset used in the evaluation is assumed Multi30K, unless indicated. Furthermore, the framework is generally evaluated on the year's MWT shared task, e.g. 2018 framework on WMT'18. Only the best results are recorded, and more comprehensive results refer to original paper.

> Nakayama et al. 2017 reported their experimental results in De2En and En2De separately, which are 13.6 and 14.1 respectively, and in table keeps the unweighted average. Chen et al. 2019 also reported their experimental results in De2En and En2De separately, which are 23.0 and 18.3 respectively, and in table keeps the unweighted average.