# **A Sociolinguistic route to the characterization and detection of the credibility of events on Twitter**

Following models are used to predict the credibility of events on twitter using socio-linguistic feature with deep learning
models.

**_Folder Description :point_left:_**
-----
 ```
- ./HAN-SL           --> Contains the code of Hierarchical attention network augmented with socio linguistic features 
                          for mutli classsification of credibiltiy.
- ./DPCNN            --> Contains the code word-level DPCNN model
- ./CharCNN          --> Contains the code for char-CNN model.
- ./Linguistic_features --> Contains the code for extracting socio-linguistc features from twitter events data.
```
**_Requirements_**
-----

Make sure to use Python3 when running the scripts. The package requirements can be obtained by running ```pip install -r requirements.txt```

**_DataSet_**
-----
We consider the largest available dataset,i.e., the CREDBANK corpus for our study. This massive dataset was constructed by iteratively tracking millions of public tweets using Twitter’s streaming API2.The corpus consists of over 66M tweets covering 1,377 events reported on Twitter between October 2014 and February 2015.
```
- Twitter stream api    : <https://developer.twitter.com/en/docs/tutorials/consuming-streaming-data.html>
- CREDBANK Corpus Paper : <https://www.aaai.org/ocs/index.php/ICWSM/ICWSM15/paper/download/10582/10509>
```
**_Models used for our this task_**
-----
We release the code for train/finetuning the following models along with their hyperparamters.
1. **HAN** : This contains basic Hierarchical attention network to detect the credibility of events.Refer to ```HAN-SL``` folder for the    codes and usage instructions.
2. **HAN-SL** : This contains basic Hierarchical attention network augment with socio-linguistic feature calculate using code in            ```Linguistic-features``` folder.Refer to ```HAN-SL``` folder for the codes and usage instructions.
3. **Bi Directional GRU** : This contains BI Direction GRU without any attention layers. Refer to ```GRU-models``` in ```HAN-SL``` for      implementation instructions.
4. **DPCNN**: This contains Deep Pyramid Convolutional Neural Networks for text categorization which we have used for our experiments      also. Refer to ```DPCNN``` folder for for the    codes and usage instructions.
5. **CharCNN**: This contains Character-level Convolutional Neural Networks for text categorization. Refer to ```Char-CNN ``` folder for    for the codes and usage instructions.

**_Blogs and github repos which we used for reference_**
-----








