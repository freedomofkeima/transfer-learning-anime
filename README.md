# Transfer Learning for Anime Characters

**Warning**: This repository size is quite big (approx. 100 MB) since it includes training and test images.

## Introduction

This repository is the continuation of [Flag #15 - Image Recognition for Anime Characters](http://freedomofkeima.com/blog/posts/flag-15-image-recognition-for-anime-characters).

In Flag #15, we can see that Transfer Learning works really well with 3 different anime characters: Nishikino Maki, Kotori Minami, and Ayase Eli.

![](screenshots/first_3_characters.png)

In this experiment, we will try to push Transfer Learning further, by using 3 different anime characters which have hair color similarity: Nishikino Maki, Takimoto Hifumi, and Sakurauchi Riko.

![](screenshots/current_3_characters.png)

This experiment has 3 main steps:
1. Utilize `lbpcascade_animeface` to recognize character face from each images
2. Resize each images to 96 x 96 pixels
3. Split images into training & test before creating the final model

![](screenshots/schema.png)

`raw` directory contains 36 images for each characters (JPG & PNG format). The first 30 images are used for training while the last 6 images are used for test.

![](screenshots/takimoto_hifumi_raw.png)

As an example, we got the following result after applying Step 1 (`cropped` directory is shown at the right side):

![](screenshots/process.png)

`lbpcascade_animeface` can detect character faces with an accuracy of around **83%**. Failed images are stored in `raw (unrecognized)` for future improvements.

Since we have 3 characters and 6 test images for each which are not part of training, `resized_for_test` contains 18 images in total. Surprisingly, almost **all characters** are detected properly!

**Update (Nov 13, 2017)**: See `animeface-2009` section below, which push face detection accuracy to **93**%.

## Requirements

- OpenCV (https://github.com/opencv/opencv)
- TensorFlow (https://github.com/tensorflow/tensorflow)

## Steps

1. The following command is used to populate `cropped` directory.

```
$ python bulk_convert.py raw/[character_name] cropped
```

2. The following command is used to populate `resized_for_training` & `resized_for_test` directory.

```
$ python bulk_resize.py cropped/[character_name] resized
```

After running the step above, you can decide how many images will be used in `resized_for_training` and how many images will be used in `resized_for_test`.

3. Re-train the Inception model by using transfer learning:

```
$ bazel-bin/tensorflow/examples/image_retraining/retrain --image_dir ~/transfer-learning-anime/resized_for_traning/
$ bazel build tensorflow/examples/image_retraining:label_image
```

4. At this point, the model is ready to use. We can run the following command to get the classification result:

```
$ bazel-bin/tensorflow/examples/image_retraining/label_image --graph=/tmp/output_graph.pb --labels=/tmp/output_labels.txt --output_layer=final_result:0 --image=$HOME/transfer-learning-anime/resized_for_test/[character name]/[image name]
```

If everything works properly, you will get the classification result.  See [TensorFlow Documentation](https://www.tensorflow.org/tutorials/image_retraining) for more options.

Optionally, sample model can be downloaded by running `download_model.sh` script inside  `models (example)` directory.

## Result Analysis

Initially, we run the experiment with 2 characters: Nishikino Maki and Takimoto Hifumi.

```
INFO:tensorflow:2017-11-10 08:50:36.151387: Step 3999: Train accuracy = 100.0%
INFO:tensorflow:2017-11-10 08:50:36.151592: Step 3999: Cross entropy = 0.002191
INFO:tensorflow:2017-11-10 08:50:36.210147: Step 3999: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:Final test accuracy = 92.9% (N=14)
```

The result is as the following:

|Image|Classification| OK/NG |
| --- | --- | --- |
| ![](resized_for_test/nishikino_maki/31.jpg) | nishikino maki (score = 0.99874)<br>takimoto hifumi (score = 0.00126) | OK |
| ![](resized_for_test/nishikino_maki/32.jpg) | nishikino maki (score = 0.75519)<br>takimoto hifumi (score = 0.24481) | OK |
| ![](resized_for_test/nishikino_maki/33.jpg) | nishikino maki (score = 0.99513)<br>takimoto hifumi (score = 0.00487) | OK |
| ![](resized_for_test/nishikino_maki/34.jpg) | nishikino maki (score = 0.98629)<br>takimoto hifumi (score = 0.01371) | OK |
| ![](resized_for_test/nishikino_maki/35.jpg) | nishikino maki (score = 0.99723)<br>takimoto hifumi (score = 0.00277) | OK |
| ![](resized_for_test/nishikino_maki/36.jpg) | nishikino maki (score = 0.99695)<br>takimoto hifumi (score = 0.00305) | OK |

|Image|Classification| OK/NG |
| --- | --- | --- |
| ![](resized_for_test/takimoto_hifumi/31.jpg) | takimoto hifumi (score = 0.63084)<br>nishikino maki (score = 0.36916) | OK|
| ![](resized_for_test/takimoto_hifumi/32.jpg) | takimoto hifumi (score = 0.99728)<br>nishikino maki (score = 0.00272) | OK |
| ![](resized_for_test/takimoto_hifumi/33.jpg) | takimoto hifumi (score = 0.99972)<br>nishikino maki (score = 0.00028) | OK |
| ![](resized_for_test/takimoto_hifumi/34.jpg) | takimoto hifumi (score = 0.98852)<br>nishikino maki (score = 0.01148) | OK |
| ![](resized_for_test/takimoto_hifumi/35.jpg) | takimoto hifumi (score = 0.99456)<br>nishikino maki (score = 0.00544) | OK |
| ![](resized_for_test/takimoto_hifumi/36.jpg) | takimoto hifumi (score = 0.96630)<br>nishikino maki (score = 0.03370) | OK |

From the result above, 10 out of 12 have threshold > 0.95, while the lowest threshold is 0.63.

At this point, I decided to add Sakurauchi Riko, which is known for its similarity to Nishikino Maki.

```
INFO:tensorflow:2017-11-10 13:13:59.270717: Step 3999: Train accuracy = 100.0%
INFO:tensorflow:2017-11-10 13:13:59.270912: Step 3999: Cross entropy = 0.005526
INFO:tensorflow:2017-11-10 13:13:59.328139: Step 3999: Validation accuracy = 100.0% (N=100)
INFO:tensorflow:Final test accuracy = 80.0% (N=15)
```

With 3 similar characters, the result is as the following:

|Image|Classification| OK/NG |
| --- | --- | --- |
| ![](resized_for_test/nishikino_maki/31.jpg) | nishikino maki (score = 0.99352)<br>sakurauchi riko (score = 0.00612)<br>takimoto hifumi (score = 0.00036) | OK |
| ![](resized_for_test/nishikino_maki/32.jpg) | nishikino maki (score = 0.47391)<br>sakurauchi riko (score = 0.37913)<br>takimoto hifumi (score = 0.14696) | OK |
| ![](resized_for_test/nishikino_maki/33.jpg) | nishikino maki (score = 0.95976)<br>sakurauchi riko (score = 0.02797)<br>takimoto hifumi (score = 0.01227) | OK |
| ![](resized_for_test/nishikino_maki/34.jpg) | nishikino maki (score = 0.88851)<br>sakurauchi riko (score = 0.07526)<br>takimoto hifumi (score = 0.03623) | OK |
| ![](resized_for_test/nishikino_maki/35.jpg) | nishikino maki (score = 0.99025)<br>sakurauchi riko (score = 0.00766)<br>takimoto hifumi (score = 0.00209) | OK |
| ![](resized_for_test/nishikino_maki/36.jpg) | nishikino maki (score = 0.96782)<br>sakurauchi riko (score = 0.02783)<br>takimoto hifumi (score = 0.00435) | OK |

As you can see above, the similarity between Nishikino Maki and Sakurauchi Miko starts to lower down the confidence level of the resulted model. Nevertheless, all classifications are still correct, where 4 out of 6 maintain the threshold of > 0.95.

|Image|Classification| OK/NG |
| --- | --- | --- |
| ![](resized_for_test/takimoto_hifumi/31.jpg) | takimoto hifumi (score = 0.86266)<br>nishikino maki (score = 0.13632)<br>sakurauchi riko (score = 0.00102) | OK |
| ![](resized_for_test/takimoto_hifumi/32.jpg) | takimoto hifumi (score = 0.87614)<br>sakurauchi riko (score = 0.12334)<br>nishikino maki (score = 0.00051) | OK |
| ![](resized_for_test/takimoto_hifumi/33.jpg) | takimoto hifumi (score = 0.99964)<br>sakurauchi riko (score = 0.00023)<br>nishikino maki (score = 0.00013) | OK |
| ![](resized_for_test/takimoto_hifumi/34.jpg) | takimoto hifumi (score = 0.99417)<br>nishikino maki (score = 0.00472)<br>sakurauchi riko (score = 0.00110) | OK |
| ![](resized_for_test/takimoto_hifumi/35.jpg) | takimoto hifumi (score = 0.94923)<br>sakurauchi riko (score = 0.04842)<br>nishikino maki (score = 0.00235) | OK |
| ![](resized_for_test/takimoto_hifumi/36.jpg) | takimoto hifumi (score = 0.96029)<br>sakurauchi riko (score = 0.02822)<br>nishikino maki (score = 0.01150) | OK |

Interestingly, the addition of 3rd character increases the confidence level of several Takimoto Hifumi testcases (see 1st and 4th result). Overall, this character can be easily differentiated compared to the other two.

|Image|Classification| OK/NG |
| --- | --- | --- |
| ![](resized_for_test/sakurauchi_riko/31.jpg) | sakurauchi riko (score = 0.98747)<br>takimoto hifumi (score = 0.01054)<br>nishikino maki (score = 0.00199) | OK |
| ![](resized_for_test/sakurauchi_riko/32.jpg) | sakurauchi riko (score = 0.96840)<br>takimoto hifumi (score = 0.02895)<br>nishikino maki (score = 0.00265) | OK |
| ![](resized_for_test/sakurauchi_riko/33.jpg) | sakurauchi riko (score = 0.97713)<br>nishikino maki (score = 0.02167)<br>takimoto hifumi (score = 0.00119) | OK |
| ![](resized_for_test/sakurauchi_riko/34.jpg) | sakurauchi riko (score = 0.90159)<br>nishikino maki (score = 0.06989)<br>takimoto hifumi (score = 0.02852) | OK |
| ![](resized_for_test/sakurauchi_riko/35.jpg) | sakurauchi riko (score = 0.99713)<br>takimoto hifumi (score = 0.00184)<br>nishikino maki (score = 0.00103) | OK |
| ![](resized_for_test/sakurauchi_riko/36.jpg) | sakurauchi riko (score = 0.79957)<br>nishikino maki (score = 0.19310)<br>takimoto hifumi (score = 0.00733) | OK |

From this experiment, it seems that the current bottleneck is located at Step 1 (face detection), which have the overall accuracy of 83% in face detection.

## animeface-2009

[nagadomi/animeface-2009](https://github.com/nagadomi/animeface-2009) provides another method of face detection. 13 out of 21 unrecognized images are now recognized in `cropped (unrecognized)` directory.

**Current found limitations**: it seems the script requires more memory and slower to run compared to `lbpcascade_animeface.xml`.

|Image|Classification| OK/NG |
| --- | --- | --- |
| ![](resized_for_test_unrecognized/nishikino_maki/1.jpg) | nishikino maki (score = 0.99296)<br>sakurauchi riko (score = 0.00694)<br>takimoto hifumi (score = 0.00010) | OK |
| ![](resized_for_test_unrecognized/nishikino_maki/3.jpg) | nishikino maki (score = 0.93702)<br>sakurauchi riko (score = 0.04017)<br>takimoto hifumi (score = 0.02281) | OK |
| ![](resized_for_test_unrecognized/nishikino_maki/4.jpg) | nishikino maki (score = 0.99406)<br>sakurauchi riko (score = 0.00565)<br>takimoto hifumi (score = 0.00030) | OK |

|Image|Classification| OK/NG |
| --- | --- | --- |
| ![](resized_for_test_unrecognized/takimoto_hifumi/2.jpg) | takimoto hifumi (score = 0.99242)<br>nishikino maki (score = 0.00431)<br>sakurauchi riko (score = 0.00327) | OK |
| ![](resized_for_test_unrecognized/takimoto_hifumi/3.jpg) | takimoto hifumi (score = 0.99596)<br>sakurauchi riko (score = 0.00403)<br>nishikino maki (score = 0.00001) | OK |
| ![](resized_for_test_unrecognized/takimoto_hifumi/4.jpg) | takimoto hifumi (score = 0.98369)<br>sakurauchi riko (score = 0.01498)<br>nishikino maki (score = 0.00133) | OK |
| ![](resized_for_test_unrecognized/takimoto_hifumi/6.jpg) | takimoto hifumi (score = 0.99796)<br>sakurauchi riko (score = 0.00189)<br>nishikino maki (score = 0.00015) | OK |
| ![](resized_for_test_unrecognized/takimoto_hifumi/8.jpg) | takimoto hifumi (score = 0.99601)<br>nishikino maki (score = 0.00335)<br>sakurauchi riko (score = 0.00064) | OK |
| ![](resized_for_test_unrecognized/takimoto_hifumi/9.jpg) | takimoto hifumi (score = 0.99960)<br>sakurauchi riko (score = 0.00029)<br>nishikino maki (score = 0.00011) | OK |
| ![](resized_for_test_unrecognized/takimoto_hifumi/10.jpg) | takimoto hifumi (score = 0.99995)<br>nishikino maki (score = 0.00004)<br>sakurauchi riko (score = 0.00001) | OK |


|Image|Classification| OK/NG |
| --- | --- | --- |
| ![](resized_for_test_unrecognized/sakurauchi_riko/2.jpg) | sakurauchi riko (score = 0.84480)<br>nishikino maki (score = 0.12101)<br>takimoto hifumi (score = 0.03419) | OK |
| ![](resized_for_test_unrecognized/sakurauchi_riko/5.jpg) | sakurauchi riko (score = 0.94310)<br>nishikino maki (score = 0.04296)<br>takimoto hifumi (score = 0.01393) | OK |
| ![](resized_for_test_unrecognized/sakurauchi_riko/7.jpg) | sakurauchi riko (score = 0.96176)<br>takimoto hifumi (score = 0.03217)<br>nishikino maki (score = 0.00607) | OK |

Since this method gives better result in detecting anime character face and classification still works with almost the same result, the overall face detection accuracy is now around **93%**.

## License

![lbpcascade_animeface.xml](lbpcascade_animeface.xml) is created by [nagadomi/lbpcascade_animeface](https://github.com/nagadomi/lbpcascade_animeface).

Copyright for all images are owned by their respective creators.
