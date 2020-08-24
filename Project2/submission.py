#Written by z5222191 for COMP9313 Project 2

#importing all the necessary functions and modules
#Based on Lab3 test environment

from pyspark.sql import DataFrame
from pyspark.ml import Pipeline, Transformer
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.ml.feature import Tokenizer, CountVectorizer, StringIndexer
import pyspark.sql.functions as F

'''
Using the Class Selector provided in Lab 3
This is used to transform the input columns to output columns
Single pipleine
'''
class Selector(Transformer):
    def __init__(self, outputCols=['id','features', 'label']):
        self.outputCols=outputCols
        
    def _transform(self, df: DataFrame) -> DataFrame:
        return df.select(*self.outputCols)
    
'''
Task 1.1
'''
    
def base_features_gen_pipeline(input_descript_col="descript", input_category_col="category", output_feature_col="features", output_label_col="label"):
    
    '''
    Token -> Vectors -> Label -> selector/transformer -> pipeline 
    '''
    
    #tokenizing the reviews in the input
    word_tokenizer = Tokenizer(inputCol="descript", outputCol="words")
    
    #Count Vectorizing using Bag of Words model
    count_vectors = CountVectorizer(inputCol="words", outputCol="features")
    
    #Labelling data for supervised learning
    label_maker = StringIndexer(inputCol = "category", outputCol = "label")
    
    #Transformer
    selector = Selector(outputCols = ['id','features', 'label'])
    
    #constructing the data
    pipeline = Pipeline(stages=[word_tokenizer, count_vectors, label_maker, selector])
    
    return pipeline

'''
Task 1.2
'''

def gen_meta_features(training_df, nb_0, nb_1, nb_2, svm_0, svm_1, svm_2):
    '''
    basic idea of this function is to obtain the prediction from each group and append it into a list
    each model prediction is stored in a dataframe
    the unncessary columns are dropped
    the predictions of all the six models are stacked vertically using join
    this dataframe is the result for the group/ for that particular iteration
    each group prediction is stored in a dataframe
    after all the group predictions are obatined, the dataframes in group_dfs are appended horizontally using union
    '''
    
    #group_dfs stores the final df at the end of every iteration
    group_dfs = []
    for i in range(5):
        #check condition to process only relevant group rows
        condition = training_df['group'] == i
        x_train = training_df.filter(~condition).cache()
        x_test = training_df.filter(condition).cache()
        
        #predict values using the six models
        nb0_model = nb_0.fit(x_train)
        nb0_pred = nb0_model.transform(x_test)
        
        nb1_model = nb_1.fit(x_train)
        nb1_pred = nb1_model.transform(x_test)
        
        nb2_model = nb_2.fit(x_train)
        nb2_pred = nb2_model.transform(x_test)

        svm0_model = svm_0.fit(x_train)
        svm0_pred = svm0_model.transform(x_test)
        
        svm1_model = svm_1.fit(x_train)
        svm1_pred = svm1_model.transform(x_test)
        
        svm2_model = svm_2.fit(x_train)
        svm2_pred = svm2_model.transform(x_test)
        
        
        #drop all columns that are not used
        drop0 = ['nb_prob_0','nb_raw_0']
        nb0_pred = nb0_pred.drop(*drop0)
        drop1 = ['label','group','features','label_0','label_1','label_2','nb_prob_1','nb_raw_1']
        nb1_pred = nb1_pred.drop(*drop1)
        drop2 = ['label','group','features','label_0','label_1','label_2','nb_prob_2','nb_raw_2']
        nb2_pred = nb2_pred.drop(*drop2)
        drop3 = ['label','group','features','label_0','label_1','label_2','label_0','svm_prob_0','svm_raw_0']
        svm0_pred = svm0_pred.drop(*drop3)
        drop4 = ['label','group','features','label_0','label_1','label_2','label_1','svm_prob_1','svm_raw_1']
        svm1_pred = svm1_pred.drop(*drop4)
        drop5 = ['label','group','features','label_0','label_1','label_2','label_2','svm_prob_2','svm_raw_2']
        svm2_pred = svm2_pred.drop(*drop5)
        
        
        #vertically stack/append dataframes using join on ID
        
        result = nb0_pred.join(nb1_pred, ["id"]).join(nb2_pred, ["id"]).join(svm0_pred, ["id"]).join(svm1_pred,["id"]).join(svm2_pred, ["id"])
        
        #Use nb pred and svm pred to get join pred
        '''
        if nb = 0 and svm = 0, then joint = 0
        if nb = 0 and svm = 1, then joint = 1
        if nb = 1 and svm = 0, then joint = 2
        if nb = 1 and svm = 1, then joint = 3
        '''
        result = result.withColumn(
        'joint_pred_0',
        F.when((F.col("nb_pred_0") == 0.0) & (F.col("svm_pred_0") == 0.0), 0.0)\
        .when((F.col("nb_pred_0") == 0.0) & (F.col("svm_pred_0") == 1.0), 1.0)\
        .when((F.col("nb_pred_0") == 1.0) & (F.col("svm_pred_0") == 0.0), 2.0)\
        .otherwise(3.0)
        )
        result = result.withColumn(
        'joint_pred_1',
        F.when((F.col("nb_pred_1") == 0.0) & (F.col("svm_pred_1") == 0.0), 0.0)\
        .when((F.col("nb_pred_1") == 0.0) & (F.col("svm_pred_1") == 1.0), 1.0)\
        .when((F.col("nb_pred_1") == 1.0) & (F.col("svm_pred_1") == 0.0), 2.0)\
        .otherwise(3.0)
        )
        result = result.withColumn(
        'joint_pred_2',
        F.when((F.col("nb_pred_2") == 0.0) & (F.col("svm_pred_2") == 0.0), 0.0)\
        .when((F.col("nb_pred_2") == 0.0) & (F.col("svm_pred_2") == 1.0), 1.0)\
        .when((F.col("nb_pred_2") == 1.0) & (F.col("svm_pred_2") == 0.0), 2.0)\
        .otherwise(3.0)
        )
        
        #append the dataframe into a list before next group is calculated
        group_dfs.append(result)
    
    #horizontal stacking of dataframes from all groups using union
    final_df = group_dfs[0].union(group_dfs[1]).union(group_dfs[2]).union(group_dfs[3]).union(group_dfs[4])
    final_df = final_df.orderBy('id')
    
    return final_df

'''
Task 1.3
'''

def test_prediction(test_df, base_features_pipeline_model, gen_base_pred_pipeline_model, gen_meta_feature_pipeline_model, meta_classifier):
    
    '''
    takes the test_df and processes the given pipelines to tranform the test data in steps
    '''
    
    #obtaining the base feature vectors
    testing_set = base_features_pipeline_model.transform(test_df)
    
    #binarizing the labels using label column
    
    testing_set = testing_set.withColumn('label_0', (testing_set['label'] == 0).cast(DoubleType()))
    testing_set = testing_set.withColumn('label_1', (testing_set['label'] == 1).cast(DoubleType()))
    testing_set = testing_set.withColumn('label_2', (testing_set['label'] == 2).cast(DoubleType()))
    #testing_set = gen_binary_labels(testing_set)
    
    #transform the the feature vectors for the various models - prediction
    step2 = gen_base_pred_pipeline_model.transform(testing_set)
    
    #get rid of unnecessary rows using drop
    drop_columns =['nb_prob_0','nb_prob_1','nb_prob_2','svm_prob_0','svm_prob_1','svm_prob_2','nb_raw_0','nb_raw_1','nb_raw_2','svm_raw_0','svm_raw_1','svm_raw_2']
    step2 = step2.drop(*drop_columns)
    
    #Calculate Joint_Prediction similar to Task 1.2 
    #Using With columns and function F
    #when and otherwise function
    
    '''
        if nb = 0 and svm = 0, then joint = 0
        if nb = 0 and svm = 1, then joint = 1
        if nb = 1 and svm = 0, then joint = 2
        if nb = 1 and svm = 1, then joint = 3
    '''
    
    #joint_pred_0
    step2 = step2.withColumn(
    'joint_pred_0',
    F.when((F.col("nb_pred_0") == 0.0) & (F.col("svm_pred_0") == 0.0), 0.0)\
    .when((F.col("nb_pred_0") == 0.0) & (F.col("svm_pred_0") == 1.0), 1.0)\
    .when((F.col("nb_pred_0") == 1.0) & (F.col("svm_pred_0") == 0.0), 2.0)\
    .otherwise(3.0)
    )
    
    #joint_pred_1
    step2 = step2.withColumn(
    'joint_pred_1',
    F.when((F.col("nb_pred_1") == 0.0) & (F.col("svm_pred_1") == 0.0), 0.0)\
    .when((F.col("nb_pred_1") == 0.0) & (F.col("svm_pred_1") == 1.0), 1.0)\
    .when((F.col("nb_pred_1") == 1.0) & (F.col("svm_pred_1") == 0.0), 2.0)\
    .otherwise(3.0)
    )
    
    #joint_pred_2
    step2 = step2.withColumn(
    'joint_pred_2',
    F.when((F.col("nb_pred_2") == 0.0) & (F.col("svm_pred_2") == 0.0), 0.0)\
    .when((F.col("nb_pred_2") == 0.0) & (F.col("svm_pred_2") == 1.0), 1.0)\
    .when((F.col("nb_pred_2") == 1.0) & (F.col("svm_pred_2") == 0.0), 2.0)\
    .otherwise(3.0)
    )
    
    #step3 involves generating meta features
    step3 = gen_meta_feature_pipeline_model.transform(step2)
    
    #step4 calculates the various vectors and gets the overall prediction 
    #using vector assembler and one hot encoding
    
    step4 = meta_classifier.transform(step3)
    
    #getting rid of unncessary columns
    final_drop = ['features','label_0','label_1','label_2','nb_pred_0','nb_pred_1','nb_pred_2','svm_pred_0',\
              'svm_pred_1','svm_pred_2','joint_pred_0','joint_pred_1','joint_pred_2','vec0','vec1','vec2',\
             'vec3','vec4','vec5','vec6','vec7','vec8','meta_features','rawPrediction','probability']
    
    #drop all columns but ID, label, and final_pred
    step4 = step4.drop(*final_drop)
    
    return step4
