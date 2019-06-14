from gensim.models import word2vec
import logging
import preprocess
def trainWord2Vec(filename):
  
    file = open(filename,'r');
    count = 0;
    lines = [];
    while count < 339:
        try:
            line = (file.readline());
            if line is not '':
                lines.append(line);
                line = '';
        except UnicodeDecodeError:
            continue;
        count += 1;
    trainData = [];
    
    file = open('text.txt','w');
    file.writelines(lines);
    file.close();
    targetWord = (lines[0].split('\t'))[0];
    for line in lines:
        sent = preprocess.preprocess(line);
        if sent is not None:
            trainData.append(sent);
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO);
    
    num_features = 300;
    min_word_count = 1;
    num_workers = 4;
    context = 10;
    downsampling = 1e-3;
    model = word2vec.Word2Vec(trainData, workers=num_workers, size=num_features, min_count = min_word_count, window = context, sample = downsampling);
    model.init_sims(replace=True);
    

    model_name = "modelWord2Vec_"+targetWord;
    model.save(model_name);
    

def driver(trainData):
    trainWord2Vec(trainData);

driver('Joy.train');
