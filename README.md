# scene-classification
Scene classification is a task of [AI Challenger competition](https://challenger.ai). This is my implementation with PyTorch.

### Method
I used 5 models:
- ResNet50  pretrained on places365
- SE-ResNet50 pretrained on ResNet50 above
- DenseNet161 pretrained on places365
- ResNeXt101 pretrained on ImageNet
- IncetpionV4 pretrained on ImageNet

Pretrain is really important. A good pretrained model can bring you a high baseline. I'm not good at Caffe so I just fine tuned ResNet50, DenseNet161 from Places365-CNNs(Because they have PyTorch weights). In fact ResNet152-places365(Caffe model) is also a good model for this task.

I train these models in different scale, for they have different capacity, big model with larger resolution.High resolution image contains more information, so it makes big model preform better.Another important reason is that models trained on different resolution are also **strongly complementary to each other.** So this is the main method I used in this competition.

In the ensemble stage, I added the **logits**(outputs of the fully connected layer, before softmax) directly to get the final prediction. Because I found in this way the logits can reserve more information than the softmax probability. In this way the score is aways higer than the score ensembled from softmax probability.

Also, some strategies like data augmentation are used:
- Scale jittering
- Color jittering
- Random Erasing
- Label shuffling
- Label smoothing

### Run
You need to download image data from the [competition website](https://challenger.ai/competition/scene/subject),
and put the training image data into 80 folders according to their labels,
then modify the path in file `dataset.py`. Then download pretrained model weights from:
[CSAILVision/places365](https://github.com/CSAILVision/places365),
and convert it into 80 classes output with `models.convert_pretrained_model.py`.

Check the model weights path and dataset path in `dataset.py` and `train.py`,
 then run the `train.py`:

 ```
 python train.py \
 --model_name=resnet50 \
 --optim=Adam \
 --learning_rate=1e-3 \
 --num_epochs=10
 ```
 Now with the default setting `--retrain=False`, it will only train the fully connected layer as a classifier.

 With setting `--retrain=True`, you can train all the parameters in the model, or you could modify the function `get_trainable_variables()` in `train.py`, like:
 ```python
 trainable_scope = [model.fc, model.layer4, model.layer3]
 ```
 then you can just train the part you want to.

 ## Result
 ### On Validation set:
 |Model Name|Image Size|12 Crop Size|Single Crop |MultiScale 12 Crop |
|:-:|:-:|:-:|:-:|:-:|
|ResNet50|224|[224,256,320,384]|95.44%|96.39%|
|IncetpionV4| 299 | [299, 320, 352, 384]|94.99%|96.12%|
|SE-ResNet50| 320 | [320, 352, 384, 416, 480]|95.84%|96.34%|
|ResNeXt101 | 320 | [320, 352, 384, 448, 480]|94.51%|95.66%|
|DenseNet161| 384 | [384, 416, 448, 480, 512] | 96.07%|96.67%|
|Ensemble  | --- | --- | --- | 97.388%|

### On Testset:
 |Dataset|Top 3|Rank|
|:-:|:-:|:-:|
|Test A| 97.202%|46th|
|Test B| 93.120%|36th|

 ## Requirement
 - [PyTorch >= 0.2.0](http://pytorch.org)
 - [tqdm](https://tqdm.github.io) as progress bar
 - TensorFlow (Either CPU or GPU version) to [use TensorBoard in PyTorch](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/04-utils/tensorboard). If you don't want to use tensorboard, just delete `from logger import Logger` in `train.py`.

## Reference
- [The Places365-CNNs for Scene Classification](https://github.com/CSAILVision/places365)
- [SENet implementation on PyTorch](https://github.com/moskomule/senet.pytorch)
- [Pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch)
- [Knowledge Guided Disambiguation for Large-Scale Scene Classification with Multi-Resolution CNNs](https://arxiv.org/abs/1610.01119)
- [Random-Erasing Data Augmentation](https://github.com/zhunzhong07/Random-Erasing)