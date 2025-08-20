## Offensive-text-detection
In this project I aim to show:
1. How we can train a model that is able to accurately predict toxic texts as well as the spans of toxic parts in said text.
2. How such a model could be utilized for moderation, specificaly, for a discord moderation bot.

### Data
* **Toxicity detection**: For this task we used the dataset provided in the **jigsaw toxic comment classification challenge** on kaggle. The data was provided in train and test splits, thus I chose to not combine
the two, opting for a simple split of the train data to obtain the validation data. After the split and some extra proccessing of the test data we ended up with:\
&nbsp;&nbsp;&nbsp;&nbsp;train: 135635 samples\
&nbsp;&nbsp;&nbsp;&nbsp;val: 23936 samples\
&nbsp;&nbsp;&nbsp;&nbsp;test: 63978 samples
* **Toxic span deetction**: Here we use the **heegyu/toxic-spans** dataset which was used in **SemEval-2021 Task 5: Toxic Spans Detection**. Yet again the data is pre-split into train and test, so we just obtain
the validation set through the training data. After the split the results are as follows:\
&nbsp;&nbsp;&nbsp;&nbsp;train: 8505 samples\
&nbsp;&nbsp;&nbsp;&nbsp;val: 1501 samples\
&nbsp;&nbsp;&nbsp;&nbsp;test: 1000 samples

This large imbalance between the sizes of the datasets will shape how we later train the model.

### Model
For the model, I chose DeBERTa-v3-base as the backbone and added 2 heads, one for the classification of the text into toxic or non-toxic (clf head) and the other for the classification of each token into toxic or non-toxic (tok head).
I also used dropout and pooling where needed.

### Training 
For the training there were 3 options to combat the great difference in size between the two datasets:
1. Combine the data, train only once by using a loss function that accounts for both tasks but assigns a higher weight to the span data, in order to account for the lack of samples
2. Train the clf head + back bone on the jigsaw data -> freeze backbone and train the tok head on the span data for a few epochs, before unfreezing the backbone and ajusting its weights for a few epochs. This will
allow us to obtain good spans without destroying the already trained model.
3. Combine the 2 options above, by training jintly first and then fine tuning with extra training

All 3 options should work just fine for this task, however I opted for the 2nd as I felt it was easier to manage and overall more intuitive, while giving reasonable results. Below are the learning curves for both
training sessions:\

During training (knowing that we will not need too many epochs beforehand) I chose to save the model weights at each epoch and choose the one that seemed to work the best.




  
