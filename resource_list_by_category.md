**Resource List**

- [Paper by category](#paper-by-category)
    - [Neural Machine Translation](#neural-machine-translation)
    - [Multimodal Language Models](#multimodal-language-models)
    - [Multimodal Machine Translation](#multimodal-machine-translation)
- [Datasets](#datasets)
- [Metrics](#metrics)
- [Tutorials](#tutorials)

## Paper by category

#### Neural Machine Translation

| _Year_ | _Authors_ | _Conf._ | _Title_ | _Links_ |
| :-:    | --        | --      | --      | --      |
| 2016 | Yang et al. | NAACL-HLT'16 | Hierarchical Attention Networks for Document Classification | [[pdf](https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf)] |
| 2016 | Zoph et al. | arXiv | Multi-Source Neural Translation | [[pdf](https://pdfs.semanticscholar.org/e4c3/aa6fef525c9ed4688125ef932a2afbbae851.pdf)] |
| 2017 | Vaswani et al. | NIPS'17 | Attention Is All You Need | [[pdf](https://arxiv.org/pdf/1706.03762.pdf)] [[github](https://github.com/tensorflow/tensor2tensor)] |
| 2017 | Xia et al. | NIPS'17 | Deliberation Networks: Sequence Generation Beyond One-Pass Decoding | [[pdf](https://papers.nips.cc/paper/6775-deliberation-networks-sequence-generation-beyond-one-pass-decoding.pdf)] [[github](https://github.com/ustctf/delibnet)] |
| 2018 | Miculicich et al. | EMNLP'18 | Document-Level Neural Machine Translation with Hierarchical Attention Networks | [[pdf](https://www.aclweb.org/anthology/D18-1325)] |
| 2018 | Devlin et al. | arXiv | BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | [[pdf](https://arxiv.org/pdf/1810.04805.pdf)] [[github](https://github.com/google-research/bert)] |
| 2018 | Yang et al. | NAACL-HLT'18 | Improving Neural Machine Translation with Conditional Sequence Generative Adversarial Nets | [[pdf](https://arxiv.org/pdf/1703.04887.pdf)] |
| 2018 | Wu et al. | NAACL-HLT'18 | Adversarial Neural Machine Translation | [[pdf](https://arxiv.org/pdf/1704.06933.pdf)] |
| 2019 | Dai et al. | ACL'19 | Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context | [[pdf](https://arxiv.org/pdf/1901.02860.pdf)] [[github](https://github.com/kimiyoung/transformer-xl)] |
| 2019 | Yang et al. | arXiv | XLNet: Generalized Autoregressive Pretraining for Language Understanding | [[pdf](https://arxiv.org/pdf/1906.08237.pdf)] [[github](https://github.com/zihangdai/xlnet)] |
| 2019 | Liu et al. | ACL'19 | Hierarchical Transformers for Multi-Document Summarization | [[pdf](https://www.aclweb.org/anthology/P19-1500)] [[github](https://github.com/nlpyang/hiersumm)] | 
| 2019 | Pourdamghani et al. | ACL'19 | Translating Translationese: A Two-Step Approach to Unsupervised Machine Translation | [[pdf](https://arxiv.org/pdf/1906.05683.pdf)] |
| 2019 | Zhou et al. | arXiv | Synchronous Bidirectional Neural Machine Translation | [[pdf](https://arxiv.org/pdf/1905.04847.pdf)] [[github](https://github.com/wszlong/sb-nmt)] |

#### Multimodal Language Models

| _Year_ | _Authors_ | _Conf._ | _Title_ | _Links_ |
| :-:    | --        | --      | --      | --      |
| 2011 | Jia et al. | ICCV'11 | Learning Cross-modality Similarity for Multinomial Data | [[pdf](https://people.eecs.berkeley.edu/~trevor/iccv11-mm.pdf)] | 
| 2014 | Mao et al. | arXiv | Explain Images with Multimodal Recurrent Neural Networks | [[pdf](https://arxiv.org/pdf/1410.1090.pdf)] |
| 2014 | Kiros et al. | arXiv | Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models | [[pdf](https://arxiv.org/pdf/1411.2539.pdf)] | 
| 2015 | Ma et al. | ICCV'15 | Multimodal Convolutional Neural Networks for Matching Image and Sentence | [[pdf](http://openaccess.thecvf.com/content_iccv_2015/papers/Ma_Multimodal_Convolutional_Neural_ICCV_2015_paper.pdf)] |
| 2015 | Mao et al. | ICLR'15 | Deep Captioning with Multimodal Recurrent Neural Networks (m-RNN) | [[pdf](https://arxiv.org/pdf/1412.6632.pdf)] [[github](https://github.com/mjhucla/mRNN-CR)] |
| 2016 | Yang et al. | NIPS'16 | Review Networks for Caption Generation | [[pdf](https://arxiv.org/pdf/1605.07912.pdf)] [[github](https://github.com/kimiyoung/review_net)] |
| 2016 | You et al. | CVPR'16 | Image Captioning with Semantic Attention | [[pdf](http://openaccess.thecvf.com/content_cvpr_2016/papers/You_Image_Captioning_With_CVPR_2016_paper.pdf)] |
| 2016 | Lu et al. | NIPS'16 | Hierarchical Question-Image Co-Attention for Visual Question Answering | [[pdf](https://arxiv.org/pdf/1606.00061.pdf)] [[github](https://github.com/jiasenlu/HieCoAttenVQA)] |
| 2018 | Anderson et al. | CVPR'18 | Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering | [[pdf](http://openaccess.thecvf.com/content_cvpr_2018/papers/Anderson_Bottom-Up_and_Top-Down_CVPR_2018_paper.pdf)] |
| 2018 | Nguyen et al. | CVPR'18 | Improved Fusion of Visual and Language Representations by Dense Symmetric Co-Attention for Visual Question Answering | [[pdf](https://arxiv.org/pdf/1804.00775.pdf)] |
| 2018 | Wang et al. | NAACL'18 | Object Counts! Bringing Explicit Detections Back into Image Captioning | [[pdf](https://www.aclweb.org/anthology/N18-1198/)] |
| 2019 | Qin et al. | CVPR'19 | Look Back and Predict Forward in Image Captioning | [[pdf](http://openaccess.thecvf.com/content_CVPR_2019/papers/Qin_Look_Back_and_Predict_Forward_in_Image_Captioning_CVPR_2019_paper.pdf)] |
| 2019 | Li et al. | AAAI'19 | Beyond RNNs: Positional Self-Attention with Co-Attention for Video Question Answering | [[pdf](https://pdfs.semanticscholar.org/5653/59aac8914505e6b02db05822ee63d3ffd03a.pdf?_ga=2.221376792.1635135941.1571362744-1866174129.1565321067)] |
| 2019 | Yu et al. | CVPR'19 | Deep Modular Co-Attention Networks for Visual Question Answering | [[pdf](http://openaccess.thecvf.com/content_CVPR_2019/papers/Yu_Deep_Modular_Co-Attention_Networks_for_Visual_Question_Answering_CVPR_2019_paper.pdf)] |

#### Multimodal Machine Translation

| _Year_ | _Authors_ | _Conf._ | _Title_ | _Links_ |
| :-:    | --        | --      | --      | --      |
| 2016 | Caglayan et al. | WMT'16 | Does Multimodality Help Human and Machine for Translation and Image Captioning? | [[pdf](https://arxiv.org/pdf/1605.09186.pdf)] |
| 2016 | Caglayan et al. | arXiv | Multimodal Attention for Neural Machine Translation | [[pdf](https://arxiv.org/pdf/1609.03976.pdf)] |
| 2016 | Huang et al. | WMT'16 | Attention-based Multimodal Neural Machine Translation | [[pdf](https://www.aclweb.org/anthology/W16-2360)] |
| 2017 | Nakayama et al. | arXiv | Zero-resource Machine Translation by Multimodal Encoder-decoder Network with Multimedia Pivot | [[pdf](https://arxiv.org/pdf/1611.04503.pdf)] |
| 2017 | Delbrouck et al. | ICLR'17 | Multimodal Compact Bilinear Pooling for Multimodal Neural Machine Translation | [[pdf](https://arxiv.org/pdf/1703.08084.pdf)] |
| 2017 | Lala et al. | PBML'17 | Unraveling the Contribution of Image Captioning and Neural Machine Translation for Multimodal Machine Translation | [[pdf](https://ufal.mff.cuni.cz/pbml/108/art-lala-madhyastha-wang-specia.pdf)] |
| 2017 | Chen et al. | arXiv | A Teacher-Student Framework for Zero-Resource Neural Machine Translation | [[pdf](https://arxiv.org/pdf/1705.00753.pdf)] |
| 2017 | Elliott et al. | arXiv | Imagination improves Multimodal Translation | [[pdf](https://arxiv.org/pdf/1705.04350.pdf)] |
| 2017 | Elliott et al. | WMT'17 | Findings of the Second Shared Task on Multimodal Machine Translation and Multilingual Image Description | [[pdf](https://arxiv.org/abs/1710.07177)] |
| 2017 | Calixto et al. | arXiv | Doubly-Attentive Decoder for Multi-modal Neural Machine Translation | [[pdf](https://arxiv.org/pdf/1702.01287.pdf)] [[github](https://github.com/iacercalixto/MultimodalNMT)] |
| 2017 | Libovicky et al. | ACL'17 | Attention Strategies for Multi-Source Sequence-to-Sequence Learning | [[pdf](https://aclweb.org/anthology/P17-2031)] |
| 2017 | Calixto et al. | EMNLP'17 | Incorporating Global Visual Features into Attention-Based Neural Machine Translation | [[pdf](https://www.aclweb.org/anthology/D17-1105)] |
| 2018 | Barrault et al. | WMT'18 | Findings of the Third Shared Task on Multimodal Machine Translation | [[pdf](http://statmt.org/wmt18/pdf/WMT029.pdf)] |
| 2018 | Caglayan et al. | WMT'18 | LIUM-CVC Submissions for WMT18 Multimodal Translation Task | [[pdf](http://statmt.org/wmt18/pdf/WMT065.pdf)] |
| 2018 | Gronroos et al. | WMT'18 | The MeMAD Submission to the WMT18 Multimodal Translation Task | [[pdf](http://statmt.org/wmt18/pdf/WMT066.pdf)] | 
| 2018 | Gwinnup et al. | WMT'18 | The AFRL-Ohio State WMT18 Multimodal System: Combining Visual with Traditional | [[pdf](http://statmt.org/wmt18/pdf/WMT067.pdf)] |
| 2018 | Helcl et al. | WMT'18 | CUNI System for the WMT18 Multimodal Translation Task | [[pdf](http://statmt.org/wmt18/pdf/WMT068.pdf)] | 
| 2018 | Lala et al. | WMT'18 | Sheffield Submissions for WMT18 Multimodal Translation Shared Task | [[pdf](http://statmt.org/wmt18/pdf/WMT069.pdf)] |
| 2018 | Zheng et al. | WMT'18 | Ensemble Sequence Level Training for Multimodal MT: OSU-Baidu WMT18 Multimodal Translation System Report | [[pdf](http://statmt.org/wmt18/pdf/WMT070.pdf)] |
| 2018 | Delbrouck et al. | WMT'18 | UMONS Submission for WMT18 Multimodal Translation Task | [[pdf](http://statmt.org/wmt18/pdf/WMT071.pdf)] [[github](https://github.com/jbdel/WMT18_MNMT)] |
| 2018 | Libovicky et al. | WMT'18 | Input Combination Strategies for Multi-Source Transformer Decoder | [[pdf](https://www.aclweb.org/anthology/W18-6326)] |
| 2018 | Shin et al. | WMT'18 | Multi-encoder Transformer Network for Automatic Post-Editing | [[pdf](https://www.aclweb.org/anthology/W18-6470)] |
| 2018 | Zhou et al. | ACL'18 | A Visual Attention Grounding Neural Model for Multimodal Machine Translation | [[pdf](https://www.aclweb.org/anthology/D18-1400)] |
| 2018 | Qian et al. | arXiv | Multimodal Machine Translation with Reinforcement Learning | [[pdf](https://arxiv.org/pdf/1805.02356.pdf)] |
| 2019 | Caglayan et al. | NAACL-HLT'19 | Probing the Need for Visual Context in Multimodal Machine Translation | [[pdf](https://arxiv.org/pdf/1903.08678.pdf)] |
| 2019 | Su et al. | CVPR'19 | Unsupervised Multi-modal Neural Machine Translation | [[pdf](https://arxiv.org/pdf/1811.11365.pdf)] |
| 2019 | Ive et al. | ACL'19 | Distilling Translations with Visual Awareness | [[pdf](https://arxiv.org/pdf/1906.07701.pdf)] [[github](https://github.com/ImperialNLP/MMT-Delib)] |
| 2019 | Calixto et al. | ACL'19 | Latent Variable Model for Multi-modal Translation | [[pdf](https://www.aclweb.org/anthology/P19-1642)] |
| 2019 | Chen et al. | IJCAI'19 | From Words to Sentences: A Progressive Learning Approach for Zero-resource Machine Translation with Visual Pivots | [[pdf](https://arxiv.org/pdf/1906.00872.pdf)] |
| 2019 | Hirasawa et al. | ACL'19 | Debiasing Word Embedding Improves Multimodal Machine Translation | [[pdf](https://arxiv.org/pdf/1905.10464.pdf)] |
| 2019 | Mogadala et al. | arXiv | Trends in Integration of Vision and Language Research: A Survey of Tasks, Datasets, and Methods | [[pdf](https://arxiv.org/pdf/1907.09358.pdf)] |
| 2019 | Calixto et al. | Springer | An Error Analysis for Image-based Multi-modal Neural Machine Translation | [[pdf](https://link.springer.com/content/pdf/10.1007%2Fs10590-019-09226-9.pdf)] |
| 2019 | Hirasawa et al. | arXiv | Multimodal Machine Translation with Embedding Prediction | [[pdf](https://arxiv.org/pdf/1904.00639.pdf)] [[github](https://github.com/toshohirasawa/nmtpytorch-emb-pred)] |
| 2020.01| Park et al. | WACV'20 | MHSAN: Multi-Head Self-Attention Network for Visual Semantic Embedding | [[pdf](https://arxiv.org/pdf/2001.03712.pdf)] [[repo](paper/park2020mhsan.pdf)] |

## Datasets

| _Dataset_ | _Authors_ | _Paper_ | _Links_ |
| --        | --        | --      | --      |
| Flickr30K | Young et al. | From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions | [[pdf](https://www.aclweb.org/anthology/Q14-1006)] [[web](http://shannon.cs.illinois.edu/DenotationGraph/)] |
| Flickr30K Entities | Plummer et al. | Flickr30K Entities: Collecting Region-to-Phrase Correspondences for Richer Image-to-Sentence Models | [[pdf](https://arxiv.org/pdf/1505.04870.pdf)] [[web](http://bryanplummer.com/Flickr30kEntities/)] [[github](https://github.com/BryanPlummer/flickr30k_entities)] | 
| Multi30K | Elliott et al. | Multi30K: Multilingual English-German Image Descriptions | [[pdf](https://arxiv.org/pdf/1605.00459.pdf)] [[github](https://github.com/multi30k/dataset)] |
| IAPR-TC12 | Grubinger et al. | The IAPR TC-12 Benchmark: A New Evaluation Resource for Visual Information Systems | [[pdf](http://thomas.deselaers.de/publications/papers/grubinger_lrec06.pdf)] [[web](https://www.imageclef.org/photodata)] |
| VATEX | Wang et al. | VATEX: A Large-Scale, High-Quality Multilingual Dataset for Video-and-Language Research | [[pdf](https://arxiv.org/pdf/1904.03493.pdf)] [[web](http://vatex.org/main/download.html)] |

## Metrics

| _Metric_ | _Authors_ | _Paper_ | _Links_ |
| --       | --        | --      | --      |
| BLEU | Papineni et al. | BLEU: a Method for Automatic Evaluation of Machine Translation | [[pdf](https://www.aclweb.org/anthology/P02-1040)] |
| METEOR | Banerjee et al. | METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments | [[pdf](http://www.cs.cmu.edu/~alavie/METEOR/pdf/Banerjee-Lavie-2005-METEOR.pdf)] [[web](http://www.cs.cmu.edu/~alavie/METEOR/)] |
| METEOR 1.5 | Denkowski et al. | METEOR Universal: Language Specific Translation Evaluation for Any Target Language | [[pdf](https://www.cs.cmu.edu/~alavie/METEOR/pdf/meteor-1.5.pdf)] [[web](http://www.cs.cmu.edu/~alavie/METEOR/)] | 
| TER | Snover et al. | A study of Translation Edit Rate with Targeted Human Annotation | [[pdf](https://www.cs.umd.edu/~snover/pub/amta06/ter_amta.pdf)] |

## Tutorials

| _Year_ | _Authors_ | _Title_ | _Links_ |
| :-:    | --        | --      | --      |
| 2016 | Elliott et al. | Multimodal Learning and Reasoning | [[pdf](https://github.com/MultimodalNLP/MultimodalNLP.github.io/raw/master/mlr_tutorial.pdf)] |
| 2017 | Lucia Specia | Multimodal Machine Translation | [[pdf](https://mtm2017.unbabel.com/assets/images/slides/lucia_specia.pdf)] |
| 2018 | Loic Barrault | Introduction to Multimodal Machine Translation | [[pdf](https://www.clsp.jhu.edu/wp-content/uploads/sites/75/2018/06/2018-06-22-Barrault-Multimodal-MT.pdf)] |
